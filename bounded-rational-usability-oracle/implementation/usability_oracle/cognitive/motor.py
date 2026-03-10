"""Motor execution models based on the Keystroke-Level Model (KLM).

Implements timing predictions for elementary motor actions in human-
computer interaction using the Keystroke-Level Model (Card, Moran &
Newell, 1983) and Fitts' Law (Fitts, 1954).  These models predict the
time a user needs to type text, point at targets, click, drag, scroll,
and execute multi-step interaction sequences.

References
----------
Card, S. K., Moran, T. P., & Newell, A. (1983). The Psychology of
    Human-Computer Interaction. Lawrence Erlbaum Associates.
Fitts, P. M. (1954). The information capacity of the human motor system
    in controlling the amplitude of movement. Journal of Experimental
    Psychology, 47(6), 381-391.
MacKenzie, I. S. (1992). Fitts' law as a research and design tool in
    human-computer interaction. Human-Computer Interaction, 7(1), 91-139.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


class MotorModel:
    """Keystroke-Level Model and Fitts' Law motor timing predictions.

    The KLM decomposes a task into a sequence of primitive operators—
    keystrokes (K), pointing (P), homing (H), mental preparation (M),
    and system response (R)—and sums their durations to predict total
    task time (Card, Moran & Newell, 1983).

    All times are in seconds.
    """

    # --- KLM keystroke constants (Card, Moran & Newell, 1983, Table 2) ------

    K_EXPERT: float = 0.120
    """Expert typist: 135 wpm → ~120 ms/keystroke."""

    K_GOOD: float = 0.200
    """Good typist: 90 wpm → ~200 ms/keystroke."""

    K_AVERAGE: float = 0.280
    """Average/skilled typist: 55 wpm → ~280 ms/keystroke."""

    K_POOR: float = 0.500
    """Poor (hunt-and-peck) typist: ~500 ms/keystroke."""

    K_DEFAULT: float = 0.280
    """Default keystroke time (average typist)."""

    # --- Other KLM operators ------------------------------------------------

    P_POINTING: float = 1.100
    """Mean pointing time: 1100 ms (Card, Moran & Newell, 1983).
    This is the average across a range of target sizes and distances;
    individual predictions use Fitts' Law."""

    H_HOMING: float = 0.400
    """Homing time: 400 ms to move the hand between keyboard and mouse."""

    M_MENTAL: float = 1.350
    """Mental preparation operator: 1350 ms."""

    R_SYSTEM: float = 0.100
    """Default system response time: 100 ms (parameterisable)."""

    # --- Fitts' Law parameters (MacKenzie, 1992) ----------------------------

    _FITTS_A: float = 0.050
    """Fitts' Law intercept (seconds)."""

    _FITTS_B: float = 0.150
    """Fitts' Law slope (seconds/bit)."""

    # --- Skill level mapping ------------------------------------------------

    _SKILL_MAP: dict[str, float] = {
        "expert": K_EXPERT,
        "good": K_GOOD,
        "average": K_AVERAGE,
        "poor": K_POOR,
    }

    # --- Elementary operators -----------------------------------------------

    @classmethod
    def keystroke_time(
        cls,
        key_type: str = "regular",
        skill_level: str = "average",
    ) -> float:
        """Return the time for a single keystroke.

        Parameters
        ----------
        key_type : str
            One of ``"regular"``, ``"shift_key"``, ``"function_key"``,
            ``"modifier_combo"``.  Modified keys take longer due to the
            additional coordination required.
        skill_level : str
            Typist proficiency: ``"expert"``, ``"good"``, ``"average"``,
            or ``"poor"``.

        Returns
        -------
        float
            Keystroke duration in seconds.
        """
        base = cls._SKILL_MAP.get(skill_level, cls.K_DEFAULT)

        # Overhead multipliers for non-regular keys
        multipliers = {
            "regular": 1.0,
            "shift_key": 1.25,       # Simultaneous Shift press
            "function_key": 1.10,    # Reaching for function row
            "modifier_combo": 1.50,  # e.g., Ctrl+Shift+K
        }
        multiplier = multipliers.get(key_type, 1.0)
        return base * multiplier

    @classmethod
    def click_time(cls) -> float:
        """Time for a single mouse click.

        Returns
        -------
        float
            Click duration (~200 ms).
        """
        return 0.200

    @classmethod
    def double_click_time(cls) -> float:
        """Time for a double-click.

        Returns
        -------
        float
            Double-click duration (~400 ms).
        """
        return 0.400

    @classmethod
    def _fitts_time(cls, distance: float, width: float) -> float:
        """Fitts' Law pointing time (Shannon formulation).

        MT = a + b * log2(distance / width + 1)

        Parameters
        ----------
        distance : float
            Movement amplitude (pixels).
        width : float
            Target width along the movement axis (pixels).

        Returns
        -------
        float
            Movement time in seconds.
        """
        distance = max(1.0, float(distance))
        width = max(1.0, float(width))
        index_of_difficulty = math.log2(distance / width + 1.0)
        return cls._FITTS_A + cls._FITTS_B * index_of_difficulty

    @classmethod
    def drag_time(
        cls,
        distance: float,
        precision: float = 1.0,
    ) -> float:
        """Time for a mouse drag operation.

        Drag comprises holding the button, moving, and releasing.
        Movement time follows Fitts' Law with an overhead for the
        sustained button press.

        Parameters
        ----------
        distance : float
            Drag distance in pixels.
        precision : float
            Effective target width at the drop location (pixels).
            Smaller values mean higher precision requirements.

        Returns
        -------
        float
            Drag duration in seconds.
        """
        precision = max(1.0, float(precision))
        drag_overhead = 0.100  # button-down + button-up
        fitts = cls._fitts_time(distance, precision)
        return drag_overhead + fitts

    @classmethod
    def scroll_time(
        cls,
        distance_pixels: float,
        velocity_pps: float = 1000.0,
    ) -> float:
        """Time to scroll by a given pixel distance.

        Parameters
        ----------
        distance_pixels : float
            Total scroll distance (pixels, positive = either direction).
        velocity_pps : float
            Scroll velocity in pixels per second.

        Returns
        -------
        float
            Scroll duration in seconds.
        """
        distance_pixels = abs(float(distance_pixels))
        velocity_pps = max(1.0, float(velocity_pps))
        # Small adjustment time for initiating and stopping scroll
        adjustment = 0.200
        return distance_pixels / velocity_pps + adjustment

    @classmethod
    def homing_time(cls) -> float:
        """Time to move the hand between keyboard and mouse.

        Returns
        -------
        float
            Homing duration in seconds (400 ms).
        """
        return cls.H_HOMING

    @classmethod
    def mental_preparation_time(cls) -> float:
        """Time for the mental preparation operator M.

        Accounts for the cognitive overhead of deciding what to do
        next in a task sequence (Card, Moran & Newell, 1983).

        Returns
        -------
        float
            Mental preparation duration in seconds (1350 ms).
        """
        return cls.M_MENTAL

    @classmethod
    def system_response_time(cls, default: float = 0.100) -> float:
        """Parameterisable system response time.

        Parameters
        ----------
        default : float
            Expected system response latency in seconds.

        Returns
        -------
        float
            System response duration.
        """
        return max(0.0, float(default))

    @classmethod
    def gesture_time(
        cls,
        complexity: float,
        n_segments: int,
    ) -> float:
        """Time for a multi-touch or stroke gesture.

        Gesture time scales with the number of movement segments and
        a complexity factor that captures curvature and precision
        requirements.  Based on extensions of Fitts' Law to stroke
        gestures (Cao & Zhai, 2007).

        Parameters
        ----------
        complexity : float
            Gesture complexity index (>= 1.0).  Higher values indicate
            more intricate shapes.
        n_segments : int
            Number of distinct movement segments in the gesture.

        Returns
        -------
        float
            Gesture execution time in seconds.
        """
        complexity = max(1.0, float(complexity))
        n_segments = max(1, int(n_segments))
        # Base segment time ~150 ms, scaled by complexity
        segment_time = 0.150 * complexity
        # Inter-segment pauses ~50 ms
        pause_time = 0.050 * max(0, n_segments - 1)
        return segment_time * n_segments + pause_time

    # --- Composite predictions ----------------------------------------------

    @classmethod
    def typing_time(
        cls,
        text: str,
        skill_level: str = "average",
    ) -> float:
        """Predict total time to type a string.

        Each character is mapped to an appropriate key type.  Upper-case
        letters and common symbols requiring Shift are classified as
        ``"shift_key"``; all others are ``"regular"``.

        Parameters
        ----------
        text : str
            The string to type.
        skill_level : str
            Typist proficiency.

        Returns
        -------
        float
            Total typing time in seconds.
        """
        if not text:
            return 0.0

        shift_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ~!@#$%^&*()_+{}|:"<>?')
        total = 0.0
        for ch in text:
            if ch in shift_chars:
                total += cls.keystroke_time("shift_key", skill_level)
            else:
                total += cls.keystroke_time("regular", skill_level)
        return total

    @classmethod
    def klm_sequence(
        cls,
        operators: list[str],
        skill_level: str = "average",
        system_response: float = 0.100,
    ) -> float:
        """Compute total execution time for a KLM operator sequence.

        Accepts a list of operator codes and sums their durations.

        Operator codes
        ^^^^^^^^^^^^^^
        - ``"K"`` — keystroke (regular)
        - ``"Ks"`` — shift keystroke
        - ``"Kf"`` — function key
        - ``"Km"`` — modifier combo
        - ``"P"`` — pointing (mean)
        - ``"H"`` — homing
        - ``"M"`` — mental preparation
        - ``"R"`` — system response
        - ``"C"`` — click
        - ``"D"`` — double-click

        Parameters
        ----------
        operators : list[str]
            Sequence of operator codes.
        skill_level : str
            Typist proficiency (for K variants).
        system_response : float
            System response time for R operators.

        Returns
        -------
        float
            Predicted total task time in seconds.
        """
        dispatch = {
            "K": lambda: cls.keystroke_time("regular", skill_level),
            "Ks": lambda: cls.keystroke_time("shift_key", skill_level),
            "Kf": lambda: cls.keystroke_time("function_key", skill_level),
            "Km": lambda: cls.keystroke_time("modifier_combo", skill_level),
            "P": lambda: cls.P_POINTING,
            "H": lambda: cls.homing_time(),
            "M": lambda: cls.mental_preparation_time(),
            "R": lambda: cls.system_response_time(system_response),
            "C": lambda: cls.click_time(),
            "D": lambda: cls.double_click_time(),
        }

        total = 0.0
        for op in operators:
            handler = dispatch.get(op)
            if handler is not None:
                total += handler()
            else:
                raise ValueError(
                    f"Unknown KLM operator '{op}'. "
                    f"Valid operators: {sorted(dispatch.keys())}"
                )
        return total
