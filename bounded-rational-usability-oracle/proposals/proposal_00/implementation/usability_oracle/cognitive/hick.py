"""Hick-Hyman law implementation with interval-arithmetic support.

Provides predictions for choice reaction time given the number of
equiprobable alternatives or arbitrary probability distributions, plus
utilities for entropy, information gain, and practice effects.

References
----------
Hick, W. E. (1952). On the rate of gain of information. *Quarterly
    Journal of Experimental Psychology*, 4(1), 11-26.
Hyman, R. (1953). Stimulus information as a determinant of reaction
    time. *Journal of Experimental Psychology*, 45(3), 188-196.
Newell, A. & Rosenbloom, P. S. (1981). Mechanisms of skill acquisition
    and the law of practice. In J. R. Anderson (Ed.), *Cognitive Skills
    and Their Acquisition* (pp. 1-55). Lawrence Erlbaum.
Seow, S. C. (2005). Information theoretic models of HCI: a comparison
    of the Hick-Hyman law and Fitts' law. *Human-Computer Interaction*,
    20(3), 315-352.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Union

import numpy as np

from usability_oracle.interval.interval import Interval


class HickHymanLaw:
    """Hick-Hyman law predictor for choice reaction time.

    The standard model is:

    .. math::

        RT = a + b \\, \\log_2(n)

    for *n* equiprobable alternatives.  When probabilities are unequal
    the generalised (Hyman, 1953) form replaces log₂(n) with the
    Shannon entropy *H* of the probability distribution.

    Default parameter values:

    * ``a = 0.200`` s (200 ms base reaction time — Hick, 1952)
    * ``b = 0.155`` s/bit (155 ms per bit of information)
    """

    DEFAULT_A: float = 0.200
    """Base reaction time (s) — Hick (1952)."""

    DEFAULT_B: float = 0.155
    """Slope (s/bit) — Hick (1952)."""

    # ------------------------------------------------------------------ #
    # Core prediction — equiprobable alternatives
    # ------------------------------------------------------------------ #

    @staticmethod
    def predict(
        n_alternatives: int,
        a: float = DEFAULT_A,
        b: float = DEFAULT_B,
    ) -> float:
        """Predict choice reaction time for *n* equiprobable stimuli.

        .. math::

            RT = a + b \\, \\log_2(n)

        Parameters
        ----------
        n_alternatives : int
            Number of stimulus-response alternatives (≥ 1).
        a : float, optional
            Base reaction time in seconds (default 0.200).
        b : float, optional
            Slope in seconds per bit (default 0.155).

        Returns
        -------
        float
            Predicted reaction time in seconds.

        Raises
        ------
        ValueError
            If *n_alternatives* < 1.
        """
        if n_alternatives < 1:
            raise ValueError(
                f"n_alternatives must be >= 1, got {n_alternatives}"
            )
        if n_alternatives == 1:
            return a  # log2(1) == 0, simple reaction time
        return a + b * math.log2(n_alternatives)

    @staticmethod
    def predict_interval(
        n: int,
        a: Interval,
        b: Interval,
    ) -> Interval:
        """Predict reaction time using interval-valued parameters.

        Parameters
        ----------
        n : int
            Number of equiprobable alternatives (≥ 1).
        a : Interval
            Intercept interval (seconds).
        b : Interval
            Slope interval (seconds/bit).

        Returns
        -------
        Interval
            Enclosing interval for the predicted reaction time.

        Raises
        ------
        ValueError
            If *n* < 1.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if n == 1:
            return a
        log2_n = Interval.from_value(math.log2(n))
        return a + b * log2_n

    # ------------------------------------------------------------------ #
    # Generalised prediction — unequal probabilities (Hyman, 1953)
    # ------------------------------------------------------------------ #

    @staticmethod
    def predict_unequal_probabilities(
        probabilities: Sequence[float],
        a: float = DEFAULT_A,
        b: float = DEFAULT_B,
    ) -> float:
        """Predict RT when alternatives have unequal probabilities.

        Uses Shannon entropy *H* in place of log₂(n):

        .. math::

            RT = a + b \\, H, \\qquad
            H = -\\sum_i p_i \\, \\log_2(p_i)

        Zero-probability alternatives are silently skipped (their
        contribution is defined as 0 by convention, since
        lim_{p→0} p log p = 0).

        Parameters
        ----------
        probabilities : sequence of float
            Probability of each alternative.  Must sum to ≈ 1.
        a : float, optional
            Base reaction time (seconds).
        b : float, optional
            Slope (seconds/bit).

        Returns
        -------
        float
            Predicted reaction time in seconds.

        Raises
        ------
        ValueError
            If probabilities do not approximately sum to 1 or contain
            negative values.
        """
        probs = np.asarray(probabilities, dtype=float)
        if np.any(probs < 0):
            raise ValueError("Probabilities must be non-negative.")
        total = float(probs.sum())
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"Probabilities must sum to 1, got {total:.8f}"
            )
        h = HickHymanLaw.entropy(probs)
        return a + b * h

    # ------------------------------------------------------------------ #
    # Information-theoretic utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def entropy(probabilities: Union[Sequence[float], np.ndarray]) -> float:
        """Compute Shannon entropy of a discrete distribution.

        .. math::

            H = -\\sum_i p_i \\, \\log_2(p_i)

        Parameters
        ----------
        probabilities : array-like of float
            Probability mass function (non-negative, should sum to 1).

        Returns
        -------
        float
            Entropy in bits (≥ 0).
        """
        probs = np.asarray(probabilities, dtype=float)
        # Mask out zeros to avoid log(0)
        mask = probs > 0
        h = -float(np.sum(probs[mask] * np.log2(probs[mask])))
        return h

    @staticmethod
    def information_gain(
        prior_probs: Sequence[float],
        posterior_probs: Sequence[float],
    ) -> float:
        """Compute Kullback-Leibler divergence D_KL(P || Q).

        .. math::

            D_{KL}(P \\| Q) = \\sum_i p_i \\, \\log_2\\!
            \\left(\\frac{p_i}{q_i}\\right)

        This measures the *information gain* when updating from the
        prior *Q* to the posterior *P*.

        Parameters
        ----------
        prior_probs : sequence of float
            Prior (reference) distribution *Q*.
        posterior_probs : sequence of float
            Posterior distribution *P*.

        Returns
        -------
        float
            KL divergence in bits (≥ 0).

        Raises
        ------
        ValueError
            If distributions have different lengths or if *Q* has a
            zero where *P* is non-zero (undefined divergence).
        """
        p = np.asarray(posterior_probs, dtype=float)
        q = np.asarray(prior_probs, dtype=float)
        if p.shape != q.shape:
            raise ValueError(
                "prior_probs and posterior_probs must have the same length."
            )
        # Where p > 0 but q == 0, KL is undefined (infinite).
        if np.any((p > 0) & (q <= 0)):
            raise ValueError(
                "KL divergence undefined: posterior non-zero where prior "
                "is zero."
            )
        mask = p > 0
        kl = float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))
        return kl

    @staticmethod
    def effective_alternatives(
        probabilities: Union[Sequence[float], np.ndarray],
    ) -> float:
        """Compute the effective number of alternatives.

        Defined as 2^H where *H* is the Shannon entropy of the
        distribution.  For a uniform distribution over *n* items this
        returns *n*.

        Parameters
        ----------
        probabilities : array-like of float
            Probability distribution.

        Returns
        -------
        float
            Effective number of alternatives (≥ 1).
        """
        h = HickHymanLaw.entropy(probabilities)
        return 2.0 ** h

    # ------------------------------------------------------------------ #
    # Practice / learning effects
    # ------------------------------------------------------------------ #

    @staticmethod
    def practice_factor(
        trials: int,
        learning_rate: float = 0.4,
    ) -> float:
        """Compute a multiplicative practice speed-up factor.

        Follows the *power law of practice* (Newell & Rosenbloom, 1981):

        .. math::

            f(n) = n^{-\\alpha}

        where *n* is the number of completed trials and α is the
        learning-rate exponent.

        Parameters
        ----------
        trials : int
            Number of practice trials completed (≥ 1).
        learning_rate : float, optional
            Power-law exponent α (default 0.4).  Larger values mean
            faster learning.

        Returns
        -------
        float
            Multiplicative factor in (0, 1] to apply to the base RT.

        Raises
        ------
        ValueError
            If *trials* < 1 or *learning_rate* < 0.
        """
        if trials < 1:
            raise ValueError(f"trials must be >= 1, got {trials}")
        if learning_rate < 0:
            raise ValueError(
                f"learning_rate must be >= 0, got {learning_rate}"
            )
        return float(trials ** (-learning_rate))

    @staticmethod
    def predict_with_practice(
        n_alternatives: int,
        trials: int,
        a: float = DEFAULT_A,
        b: float = DEFAULT_B,
        learning_rate: float = 0.4,
    ) -> float:
        """Predict RT accounting for the power law of practice.

        Combines :meth:`predict` with :meth:`practice_factor`:

        .. math::

            RT_{\\text{practised}} = f(n_{\\text{trials}}) \\,
            \\bigl(a + b \\, \\log_2(n)\\bigr)

        Parameters
        ----------
        n_alternatives : int
            Number of equiprobable alternatives (≥ 1).
        trials : int
            Number of prior practice trials (≥ 1).
        a : float, optional
            Base reaction time (seconds).
        b : float, optional
            Slope (seconds/bit).
        learning_rate : float, optional
            Power-law exponent (default 0.4).

        Returns
        -------
        float
            Practice-adjusted predicted reaction time (seconds).
        """
        base_rt = HickHymanLaw.predict(n_alternatives, a, b)
        factor = HickHymanLaw.practice_factor(trials, learning_rate)
        return base_rt * factor

    # ------------------------------------------------------------------ #
    # Stimulus-response compatibility
    # ------------------------------------------------------------------ #

    @staticmethod
    def stimulus_response_compatibility(
        n_alternatives: int,
        compatibility_factor: float = 1.0,
    ) -> float:
        """Adjust the Hick-Hyman slope for S-R compatibility.

        High stimulus-response compatibility (e.g. spatially congruent
        key mappings) reduces the effective information-processing rate
        and thereby the slope *b*.

        .. math::

            b_{\\text{eff}} = b \\, / \\, c

        where *c* is the compatibility factor (≥ 1 means compatible,
        values < 1 mean *incompatible* mappings that slow the response).

        Parameters
        ----------
        n_alternatives : int
            Number of equiprobable alternatives (≥ 1).
        compatibility_factor : float, optional
            Compatibility multiplier (default 1.0 = neutral).  Values
            > 1 speed up responses; values in (0, 1) slow them down.

        Returns
        -------
        float
            Adjusted predicted reaction time (seconds).

        Raises
        ------
        ValueError
            If *compatibility_factor* ≤ 0.
        """
        if compatibility_factor <= 0:
            raise ValueError(
                f"compatibility_factor must be > 0, got {compatibility_factor}"
            )
        effective_b = HickHymanLaw.DEFAULT_B / compatibility_factor
        return HickHymanLaw.predict(
            n_alternatives,
            a=HickHymanLaw.DEFAULT_A,
            b=effective_b,
        )
