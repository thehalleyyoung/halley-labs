"""Functional interface to interval arithmetic operations.

Provides :class:`IntervalArithmetic` — a collection of static methods
that mirror the operators on :class:`~usability_oracle.interval.interval.Interval`
but expose a purely functional API suitable for use in expression-tree
evaluators and propagation engines.

All methods accept and return :class:`Interval` instances.

References
----------
Moore, R. E., Kearfott, R. B., & Cloud, M. J. (2009).
    *Introduction to Interval Analysis*. SIAM.
Neumaier, A. (1990).
    *Interval Methods for Systems of Equations*. Cambridge University Press.
"""

from __future__ import annotations

from typing import Sequence

from usability_oracle.interval.interval import Interval


class IntervalArithmetic:
    """Static-method collection for interval arithmetic.

    Every method delegates to the corresponding :class:`Interval`
    operator but provides a uniform, named-function interface.
    """

    # ------------------------------------------------------------------
    # Basic binary operations
    # ------------------------------------------------------------------

    @staticmethod
    def add(a: Interval, b: Interval) -> Interval:
        """Interval addition.

        .. math::
            [a, b] + [c, d] = [a + c,\\; b + d]

        Parameters
        ----------
        a, b : Interval
            Operands.

        Returns
        -------
        Interval
            Tightest enclosure of the sum.
        """
        return a + b

    @staticmethod
    def subtract(a: Interval, b: Interval) -> Interval:
        """Interval subtraction.

        .. math::
            [a, b] - [c, d] = [a - d,\\; b - c]

        Parameters
        ----------
        a, b : Interval
            Operands.

        Returns
        -------
        Interval
            Tightest enclosure of the difference.
        """
        return a - b

    @staticmethod
    def multiply(a: Interval, b: Interval) -> Interval:
        """Interval multiplication.

        Uses the sign-classified four-product method implemented in
        :meth:`Interval.__mul__`.

        Parameters
        ----------
        a, b : Interval
            Operands.

        Returns
        -------
        Interval
            Tightest enclosure of the product.
        """
        return a * b

    @staticmethod
    def divide(a: Interval, b: Interval) -> Interval:
        """Interval division.

        Raises :class:`ZeroDivisionError` when *b* contains zero in its
        interior.

        Parameters
        ----------
        a : Interval
            Numerator interval.
        b : Interval
            Denominator interval (must not straddle zero).

        Returns
        -------
        Interval
            Tightest enclosure of the quotient.
        """
        return a / b

    # ------------------------------------------------------------------
    # Power / root / exponential / logarithm
    # ------------------------------------------------------------------

    @staticmethod
    def power(a: Interval, n: int) -> Interval:
        """Raise interval *a* to non-negative integer power *n*.

        Odd powers preserve monotonicity; even powers must account for
        sign changes when the interval straddles zero.

        Parameters
        ----------
        a : Interval
            Base interval.
        n : int
            Non-negative integer exponent.

        Returns
        -------
        Interval
        """
        return a ** n

    @staticmethod
    def sqrt(a: Interval) -> Interval:
        """Element-wise square root of a non-negative interval.

        .. math::
            \\sqrt{[a, b]} = [\\sqrt{a},\\; \\sqrt{b}]

        Parameters
        ----------
        a : Interval
            Must have ``a.low >= 0``.

        Returns
        -------
        Interval
        """
        return a.sqrt()

    @staticmethod
    def log(a: Interval) -> Interval:
        """Natural logarithm of a strictly positive interval.

        .. math::
            \\ln([a, b]) = [\\ln(a),\\; \\ln(b)]

        Parameters
        ----------
        a : Interval
            Must have ``a.low > 0``.

        Returns
        -------
        Interval
        """
        return a.log()

    @staticmethod
    def exp(a: Interval) -> Interval:
        """Element-wise exponential of an interval.

        .. math::
            \\exp([a, b]) = [\\exp(a),\\; \\exp(b)]

        Parameters
        ----------
        a : Interval

        Returns
        -------
        Interval
        """
        return a.exp()

    # ------------------------------------------------------------------
    # Min / Max
    # ------------------------------------------------------------------

    @staticmethod
    def min_interval(a: Interval, b: Interval) -> Interval:
        """Interval enclosure of min(x, y) for x ∈ *a*, y ∈ *b*.

        .. math::
            \\min([a, b], [c, d]) = [\\min(a, c),\\; \\min(b, d)]

        Parameters
        ----------
        a, b : Interval

        Returns
        -------
        Interval
        """
        return Interval(min(a.low, b.low), min(a.high, b.high))

    @staticmethod
    def max_interval(a: Interval, b: Interval) -> Interval:
        """Interval enclosure of max(x, y) for x ∈ *a*, y ∈ *b*.

        .. math::
            \\max([a, b], [c, d]) = [\\max(a, c),\\; \\max(b, d)]

        Parameters
        ----------
        a, b : Interval

        Returns
        -------
        Interval
        """
        return Interval(max(a.low, b.low), max(a.high, b.high))

    # ------------------------------------------------------------------
    # Aggregate operations
    # ------------------------------------------------------------------

    @staticmethod
    def sum_intervals(intervals: Sequence[Interval]) -> Interval:
        """Sum a sequence of intervals.

        Computes :math:`\\sum_{i} X_i` using left-to-right accumulation.

        Parameters
        ----------
        intervals : Sequence[Interval]
            Non-empty sequence.

        Returns
        -------
        Interval
            Tightest enclosure of the sum.

        Raises
        ------
        ValueError
            If *intervals* is empty.
        """
        if not intervals:
            raise ValueError("Cannot sum an empty sequence of intervals.")
        result = intervals[0]
        for iv in intervals[1:]:
            result = result + iv
        return result

    @staticmethod
    def dot_product(
        a: Sequence[Interval], b: Sequence[Interval]
    ) -> Interval:
        """Interval dot product of two equal-length vectors.

        .. math::
            \\mathbf{a} \\cdot \\mathbf{b} = \\sum_i a_i \\, b_i

        Parameters
        ----------
        a, b : Sequence[Interval]
            Equal-length sequences.

        Returns
        -------
        Interval

        Raises
        ------
        ValueError
            If the sequences have different lengths or are empty.
        """
        if len(a) != len(b):
            raise ValueError(
                f"Vectors must have equal length, got {len(a)} and {len(b)}."
            )
        if not a:
            raise ValueError("Cannot compute dot product of empty vectors.")
        products = [ai * bi for ai, bi in zip(a, b)]
        result = products[0]
        for p in products[1:]:
            result = result + p
        return result

    @staticmethod
    def weighted_sum(
        values: Sequence[Interval], weights: Sequence[Interval]
    ) -> Interval:
        """Weighted sum of interval values with interval weights.

        .. math::
            \\sum_i w_i \\, v_i

        This is mathematically identical to :meth:`dot_product` but is
        provided for semantic clarity in model-evaluation contexts where
        one vector represents uncertain parameters and the other
        represents uncertain weights.

        Parameters
        ----------
        values : Sequence[Interval]
            Uncertain parameter values.
        weights : Sequence[Interval]
            Uncertain weights.

        Returns
        -------
        Interval
        """
        return IntervalArithmetic.dot_product(weights, values)
