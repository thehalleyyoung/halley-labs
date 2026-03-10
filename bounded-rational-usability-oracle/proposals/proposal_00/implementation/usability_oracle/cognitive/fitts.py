"""Fitts' law implementation with interval-arithmetic support.

Provides point-estimate and interval-valued predictions for aimed
movements, along with derived quantities such as throughput, effective
width, and crossing time (the steering law variant).

References
----------
Fitts, P. M. (1954). The information capacity of the human motor system
    in controlling the amplitude of movement. *Journal of Experimental
    Psychology*, 47(6), 381-391.
MacKenzie, I. S. (1992). Fitts' law as a research and design tool in
    human-computer interaction. *Human-Computer Interaction*, 7(1),
    91-139.
Card, S. K., Moran, T. P., & Newell, A. (1983).
    *The Psychology of Human-Computer Interaction*. Lawrence Erlbaum.
ISO 9241-411:2012. Ergonomics of human-system interaction — Part 411:
    Evaluation methods for the design of physical input devices.
Accot, J. & Zhai, S. (1997). Beyond Fitts' law: models for trajectory-
    based HCI tasks. *Proc. CHI '97*, 295-302.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from usability_oracle.interval.interval import Interval


class FittsLaw:
    """Fitts' law predictor for aimed-movement time.

    The Shannon formulation (MacKenzie, 1992) is used throughout:

    .. math::

        MT = a + b \\, \\log_2\\!\\left(1 + \\frac{D}{W}\\right)

    where *D* is the centre-to-centre distance, *W* is the target width
    along the axis of approach, *a* is the intercept (seconds), and *b*
    is the slope (seconds per bit).

    Default parameter values follow Card, Moran & Newell (1983):

    * ``a = 0.050`` s (50 ms intercept)
    * ``b = 0.150`` s/bit (150 ms per bit)
    """

    DEFAULT_A: float = 0.050
    """Intercept (s) — Card, Moran & Newell (1983)."""

    DEFAULT_B: float = 0.150
    """Slope (s/bit) — Card, Moran & Newell (1983)."""

    # ------------------------------------------------------------------ #
    # Core prediction
    # ------------------------------------------------------------------ #

    @staticmethod
    def predict(
        distance: float,
        width: float,
        a: float = DEFAULT_A,
        b: float = DEFAULT_B,
    ) -> float:
        """Predict movement time for a single aimed movement.

        Parameters
        ----------
        distance : float
            Centre-to-centre distance to the target (pixels, > 0).
        width : float
            Target width along the movement axis (pixels, > 0).
        a : float, optional
            Intercept in seconds (default 0.050).
        b : float, optional
            Slope in seconds per bit (default 0.150).

        Returns
        -------
        float
            Predicted movement time in seconds.

        Raises
        ------
        ValueError
            If *distance* or *width* is not positive.
        """
        if distance <= 0:
            raise ValueError(f"distance must be > 0, got {distance}")
        if width <= 0:
            raise ValueError(f"width must be > 0, got {width}")
        return a + b * math.log2(1.0 + distance / width)

    @staticmethod
    def predict_interval(
        distance: Interval,
        width: Interval,
        a: Interval,
        b: Interval,
    ) -> Interval:
        """Predict movement time using interval arithmetic.

        All inputs are :class:`Interval` objects so that parameter
        uncertainty and geometric tolerances propagate naturally into
        the output bound.

        Parameters
        ----------
        distance : Interval
            Distance interval (both bounds > 0).
        width : Interval
            Width interval (both bounds > 0).
        a : Interval
            Intercept interval (seconds).
        b : Interval
            Slope interval (seconds/bit).

        Returns
        -------
        Interval
            Enclosing interval for the predicted movement time.
        """
        one = Interval.from_value(1.0)
        ratio = distance / width
        id_interval = (one + ratio).log2()
        return a + b * id_interval

    # ------------------------------------------------------------------ #
    # Derived quantities
    # ------------------------------------------------------------------ #

    @staticmethod
    def index_of_difficulty(distance: float, width: float) -> float:
        """Compute the index of difficulty (Shannon formulation).

        .. math::

            ID = \\log_2\\!\\left(1 + \\frac{D}{W}\\right)

        Parameters
        ----------
        distance : float
            Centre-to-centre distance (pixels, > 0).
        width : float
            Target width (pixels, > 0).

        Returns
        -------
        float
            Index of difficulty in bits.

        Raises
        ------
        ValueError
            If *distance* or *width* is not positive.
        """
        if distance <= 0:
            raise ValueError(f"distance must be > 0, got {distance}")
        if width <= 0:
            raise ValueError(f"width must be > 0, got {width}")
        return math.log2(1.0 + distance / width)

    @staticmethod
    def throughput(
        distance: float,
        width: float,
        movement_time: float,
    ) -> float:
        """Compute throughput (bits/s) for one trial.

        .. math::

            TP = \\frac{ID}{MT}

        Parameters
        ----------
        distance : float
            Centre-to-centre distance (pixels, > 0).
        width : float
            Target width (pixels, > 0).
        movement_time : float
            Observed movement time in seconds (> 0).

        Returns
        -------
        float
            Throughput in bits per second.

        Raises
        ------
        ValueError
            If any argument is not positive.
        """
        if movement_time <= 0:
            raise ValueError(
                f"movement_time must be > 0, got {movement_time}"
            )
        idx = FittsLaw.index_of_difficulty(distance, width)
        return idx / movement_time

    # ------------------------------------------------------------------ #
    # Effective-width adjustments (ISO 9241-411)
    # ------------------------------------------------------------------ #

    @staticmethod
    def effective_width(
        hits_x: np.ndarray,
        hits_y: np.ndarray,
    ) -> float:
        """Compute effective target width from endpoint scatter.

        Uses the ISO 9241-411 convention:

        .. math::

            W_e = 4.133 \\, \\sigma_x

        where σ_x is the standard deviation of the hit coordinates
        projected onto the primary movement axis (*x*).

        Parameters
        ----------
        hits_x : numpy.ndarray
            X-coordinates of endpoint hits (≥ 2 samples).
        hits_y : numpy.ndarray
            Y-coordinates of endpoint hits (same length; used for
            validation only — the projection is along *x*).

        Returns
        -------
        float
            Effective width in the same units as *hits_x*.

        Raises
        ------
        ValueError
            If fewer than two hits are provided.
        """
        hits_x = np.asarray(hits_x, dtype=float)
        hits_y = np.asarray(hits_y, dtype=float)
        if hits_x.size < 2:
            raise ValueError("Need at least 2 hit samples.")
        if hits_x.shape != hits_y.shape:
            raise ValueError("hits_x and hits_y must have the same shape.")
        sd_x = float(np.std(hits_x, ddof=1))
        return 4.133 * sd_x

    @staticmethod
    def effective_id(distance: float, effective_width: float) -> float:
        """Compute effective index of difficulty.

        .. math::

            ID_e = \\log_2\\!\\left(1 + \\frac{D}{W_e}\\right)

        Parameters
        ----------
        distance : float
            Centre-to-centre distance (pixels, > 0).
        effective_width : float
            Effective target width from :meth:`effective_width` (> 0).

        Returns
        -------
        float
            Effective index of difficulty in bits.
        """
        return FittsLaw.index_of_difficulty(distance, effective_width)

    # ------------------------------------------------------------------ #
    # Steering law (Accot & Zhai, 1997)
    # ------------------------------------------------------------------ #

    @staticmethod
    def crossing_time(
        amplitude: float,
        tolerance: float,
        a: float = DEFAULT_A,
        b: float = DEFAULT_B,
    ) -> float:
        """Predict time to steer through a constrained tunnel.

        The *steering law* (Accot & Zhai, 1997) generalises Fitts' law
        to trajectory tasks:

        .. math::

            T = a + b \\, \\frac{A}{W}

        where *A* is the path amplitude and *W* is the tunnel width
        (tolerance).

        Parameters
        ----------
        amplitude : float
            Path length through the tunnel (> 0).
        tolerance : float
            Width of the tunnel / acceptable deviation (> 0).
        a : float, optional
            Intercept (seconds).
        b : float, optional
            Slope (seconds per unit index of difficulty).

        Returns
        -------
        float
            Predicted traversal time in seconds.

        Raises
        ------
        ValueError
            If *amplitude* or *tolerance* is not positive.
        """
        if amplitude <= 0:
            raise ValueError(f"amplitude must be > 0, got {amplitude}")
        if tolerance <= 0:
            raise ValueError(f"tolerance must be > 0, got {tolerance}")
        return a + b * (amplitude / tolerance)

    # ------------------------------------------------------------------ #
    # Batch / vectorised prediction
    # ------------------------------------------------------------------ #

    @staticmethod
    def predict_batch(
        distances: np.ndarray,
        widths: np.ndarray,
        a: float = DEFAULT_A,
        b: float = DEFAULT_B,
    ) -> np.ndarray:
        """Vectorised movement-time prediction over arrays.

        Parameters
        ----------
        distances : numpy.ndarray
            Array of distances (all > 0).
        widths : numpy.ndarray
            Array of target widths (all > 0), broadcastable with
            *distances*.
        a : float, optional
            Intercept (seconds).
        b : float, optional
            Slope (seconds per bit).

        Returns
        -------
        numpy.ndarray
            Predicted movement times (same shape as broadcast result).

        Raises
        ------
        ValueError
            If any distance or width element is ≤ 0.
        """
        distances = np.asarray(distances, dtype=float)
        widths = np.asarray(widths, dtype=float)
        if np.any(distances <= 0):
            raise ValueError("All distances must be > 0.")
        if np.any(widths <= 0):
            raise ValueError("All widths must be > 0.")
        return a + b * np.log2(1.0 + distances / widths)

    # ------------------------------------------------------------------ #
    # Error-rate estimation
    # ------------------------------------------------------------------ #

    @staticmethod
    def error_rate(
        distance: float,
        width: float,
        actual_width: Optional[float] = None,
    ) -> float:
        """Estimate miss probability from the speed-accuracy trade-off.

        Assumes endpoint scatter is normally distributed with standard
        deviation proportional to distance and inversely proportional to
        the movement time predicted by Fitts' law.  The miss probability
        is approximated as:

        .. math::

            P_{\\text{miss}} \\approx 1 - \\operatorname{erf}\\!
            \\left(\\frac{W_a}{\\sqrt{2}\\,\\sigma}\\right)

        where :math:`\\sigma = 0.5\\,W` (i.e. the nominal width
        corresponds to ≈ 96 % of endpoints for a well-aimed movement)
        and :math:`W_a` is the *actual* presented width (which may
        differ from the nominal *W* used to compute ID).

        Parameters
        ----------
        distance : float
            Centre-to-centre distance (pixels, > 0).
        width : float
            Nominal target width used for the ID computation (> 0).
        actual_width : float or None, optional
            Actual (rendered) target width.  Defaults to *width* when
            ``None``.

        Returns
        -------
        float
            Estimated error (miss) probability in [0, 1].
        """
        if distance <= 0:
            raise ValueError(f"distance must be > 0, got {distance}")
        if width <= 0:
            raise ValueError(f"width must be > 0, got {width}")
        if actual_width is None:
            actual_width = width
        if actual_width <= 0:
            raise ValueError(
                f"actual_width must be > 0, got {actual_width}"
            )
        # σ chosen so that 4.133σ ≈ W  →  σ = W / 4.133
        sigma = width / 4.133
        # Miss probability: fraction of Gaussian outside ±actual_width/2
        p_hit = math.erf(actual_width / (2.0 * sigma * math.sqrt(2.0)))
        return 1.0 - p_hit
