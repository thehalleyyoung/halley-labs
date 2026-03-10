"""Uncertainty propagation strategies for interval-valued parameters.

This module provides four complementary strategies for propagating
parameter intervals through arbitrary functions:

1. **Natural interval extension** — evaluate the function with interval
   operands.  Guaranteed to enclose the true range but may over-
   approximate due to the *dependency problem*.
2. **Monte Carlo sampling** — draw uniform samples from each parameter
   interval and compute an empirical enclosure.  Not guaranteed but
   useful for validation and for functions without closed-form interval
   extensions.
3. **Affine propagation** — exact propagation for linear (affine)
   functions :math:`f(\\mathbf{x}) = \\mathbf{c}^\\top \\mathbf{x} + d`.
4. **First-order Taylor propagation** — linearise around a centre point
   using the gradient to obtain a first-order enclosure.

References
----------
Moore, R. E., Kearfott, R. B., & Cloud, M. J. (2009).
    *Introduction to Interval Analysis*. SIAM.
Stolfi, J., & de Figueiredo, L. H. (1997).
    Self-validated numerical methods and applications. *Monograph for
    21st Brazilian Mathematics Colloquium*, IMPA.
Jaulin, L., Kieffer, M., Didrit, O., & Walter, É. (2001).
    *Applied Interval Analysis*. Springer.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from usability_oracle.interval.interval import Interval


class ErrorPropagation:
    """Strategies for propagating uncertainty through functions.

    All methods are static so that the class acts as a namespace
    rather than requiring instantiation.
    """

    # ------------------------------------------------------------------
    # Natural interval extension
    # ------------------------------------------------------------------

    @staticmethod
    def propagate_linear(
        f: Callable[..., Interval],
        intervals: Sequence[Interval],
    ) -> Interval:
        """Evaluate the *natural interval extension* of *f*.

        The callable *f* must accept :class:`Interval` arguments and
        return an :class:`Interval`.  It is invoked once with the given
        parameter intervals.

        Because :class:`Interval` overloads the arithmetic operators,
        any expression composed of ``+``, ``-``, ``*``, ``/``, and
        ``**`` (with integer exponent) will automatically produce the
        natural interval extension when called with interval operands.

        .. warning::
            The natural extension is an *enclosure* — it always contains
            the true range but may be wider due to the dependency
            problem (repeated occurrences of the same variable are
            treated independently).

        Parameters
        ----------
        f : Callable[..., Interval]
            Function written in terms of Interval-compatible operations.
        intervals : Sequence[Interval]
            One interval per parameter of *f*.

        Returns
        -------
        Interval
            Enclosure of the range of *f* over the Cartesian product
            of *intervals*.

        Examples
        --------
        >>> x = Interval(1.0, 2.0)
        >>> y = Interval(3.0, 4.0)
        >>> ErrorPropagation.propagate_linear(lambda a, b: a + b, [x, y])
        Interval(4.0, 6.0)
        """
        return f(*intervals)

    # ------------------------------------------------------------------
    # Monte Carlo
    # ------------------------------------------------------------------

    @staticmethod
    def propagate_monte_carlo(
        f: Callable[..., float],
        intervals: Sequence[Interval],
        n_samples: int = 10_000,
        *,
        seed: int | None = None,
        confidence_expand: float = 0.0,
    ) -> Interval:
        """Estimate the range of *f* via uniform Monte Carlo sampling.

        For each sample, one value is drawn uniformly at random from
        each parameter interval.  The function *f* is evaluated at each
        sampled point and the resulting minimum/maximum define the
        empirical enclosure.

        .. note::
            The result is **not** a guaranteed enclosure — it is a
            statistical estimate.  Increase *n_samples* or use
            *confidence_expand* to widen the result conservatively.

        Parameters
        ----------
        f : Callable[..., float]
            Scalar-valued function of ``len(intervals)`` real arguments.
        intervals : Sequence[Interval]
            One interval per parameter.
        n_samples : int
            Number of random evaluation points (default 10 000).
        seed : int | None
            Optional RNG seed for reproducibility.
        confidence_expand : float
            Non-negative fraction by which to expand the empirical
            interval on each side.  For example, 0.01 adds 1 % of the
            empirical width to both endpoints.

        Returns
        -------
        Interval
            Empirical enclosure of *f* over the parameter intervals.

        Raises
        ------
        ValueError
            If *n_samples* < 1 or *confidence_expand* < 0.
        """
        if n_samples < 1:
            raise ValueError(f"n_samples must be ≥ 1, got {n_samples}.")
        if confidence_expand < 0.0:
            raise ValueError(
                f"confidence_expand must be non-negative, got {confidence_expand}."
            )

        rng = np.random.default_rng(seed)

        # Build a (n_samples × n_params) matrix of uniform samples.
        n_params = len(intervals)
        lows = np.array([iv.low for iv in intervals], dtype=np.float64)
        highs = np.array([iv.high for iv in intervals], dtype=np.float64)

        samples = rng.uniform(lows, highs, size=(n_samples, n_params))

        # Evaluate f at every sample point.
        results = np.empty(n_samples, dtype=np.float64)
        for i in range(n_samples):
            results[i] = f(*samples[i])

        lo = float(np.min(results))
        hi = float(np.max(results))

        if confidence_expand > 0.0:
            w = hi - lo
            lo -= confidence_expand * w
            hi += confidence_expand * w

        return Interval(lo, hi)

    # ------------------------------------------------------------------
    # Affine (exact for linear functions)
    # ------------------------------------------------------------------

    @staticmethod
    def propagate_affine(
        coefficients: Sequence[float],
        intervals: Sequence[Interval],
        constant: float = 0.0,
    ) -> Interval:
        """Exact propagation for the affine function

        .. math::
            f(\\mathbf{x}) = c_0 x_0 + c_1 x_1 + \\cdots + c_{n-1} x_{n-1} + d

        This avoids the dependency problem entirely because each
        variable appears exactly once.

        Parameters
        ----------
        coefficients : Sequence[float]
            Coefficient vector :math:`\\mathbf{c}`.
        intervals : Sequence[Interval]
            One interval per variable.
        constant : float
            Additive constant *d* (default 0).

        Returns
        -------
        Interval
            Exact range of the affine function over the given intervals.

        Raises
        ------
        ValueError
            If *coefficients* and *intervals* differ in length.
        """
        if len(coefficients) != len(intervals):
            raise ValueError(
                f"coefficients ({len(coefficients)}) and intervals "
                f"({len(intervals)}) must have the same length."
            )

        lo = constant
        hi = constant

        for c, iv in zip(coefficients, intervals):
            c = float(c)
            if c >= 0.0:
                lo += c * iv.low
                hi += c * iv.high
            else:
                lo += c * iv.high
                hi += c * iv.low

        return Interval(lo, hi)

    # ------------------------------------------------------------------
    # First-order Taylor
    # ------------------------------------------------------------------

    @staticmethod
    def taylor_propagation(
        f: Callable[..., float],
        f_grad: Callable[..., Sequence[float]],
        center: Sequence[float],
        intervals: Sequence[Interval],
    ) -> Interval:
        """First-order Taylor enclosure of *f* around *center*.

        Linearises *f* at the point *center* using the gradient
        *f_grad* and propagates the resulting affine form exactly:

        .. math::
            f(\\mathbf{x}) \\approx f(\\mathbf{c})
                + \\nabla f(\\mathbf{c})^\\top (\\mathbf{x} - \\mathbf{c})

        The interval enclosure of the right-hand side is:

        .. math::
            \\bigl[f(\\mathbf{c})\\bigr]
                + \\sum_i \\nabla_i f(\\mathbf{c})\\;\\bigl([x_i] - c_i\\bigr)

        This is exact for affine *f* and a first-order approximation
        for smooth non-linear *f*.

        Parameters
        ----------
        f : Callable[..., float]
            Scalar function evaluated at a point (not interval).
        f_grad : Callable[..., Sequence[float]]
            Gradient of *f*.  Must return a sequence of partial
            derivatives with the same length as *center*.
        center : Sequence[float]
            Point at which to linearise (typically the midpoint of each
            parameter interval).
        intervals : Sequence[Interval]
            Parameter intervals.

        Returns
        -------
        Interval
            First-order Taylor enclosure.

        Raises
        ------
        ValueError
            If *center* and *intervals* differ in length.
        """
        if len(center) != len(intervals):
            raise ValueError(
                f"center ({len(center)}) and intervals ({len(intervals)}) "
                "must have the same length."
            )

        f_c = float(f(*center))
        grad = f_grad(*center)

        # Build shifted intervals (x_i − c_i).
        shifted: list[Interval] = []
        for c_i, iv in zip(center, intervals):
            shifted.append(Interval(iv.low - c_i, iv.high - c_i))

        # Propagate the affine form ∑ g_i · (x_i − c_i) + f(c).
        return ErrorPropagation.propagate_affine(
            [float(g) for g in grad],
            shifted,
            constant=f_c,
        )
