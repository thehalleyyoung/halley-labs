"""ACT-R motor module for usability modelling.

Implements the motor module of the ACT-R cognitive architecture
(Anderson, 2007; Byrne & Anderson, 2001), integrating Fitts' law for
movement time prediction, motor preparation and execution staging, hand
and finger tracking, and models for typing, mouse movement, and touch
interaction.

References
----------
Anderson, J. R. (2007). *How Can the Human Mind Occur in the Physical
    Universe?* Oxford University Press.
Byrne, M. D. & Anderson, J. R. (2001). Serial modules in parallel:
    The psychological refractory period and perfect time-sharing.
    *Psychological Review*, 108(4), 847-869.
Fitts, P. M. (1954). The information capacity of the human motor system
    in controlling the amplitude of movement. *Journal of Experimental
    Psychology*, 47(6), 381-391.
John, B. E. (1996). TYPIST: A theory of performance in skilled typing.
    *Human-Computer Interaction*, 11(4), 321-355.
Card, S. K., Moran, T. P., & Newell, A. (1983). *The Psychology of
    Human-Computer Interaction*. Lawrence Erlbaum.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from usability_oracle.cognitive.models import BoundingBox, Point2D


# ---------------------------------------------------------------------------
# Hand / finger state
# ---------------------------------------------------------------------------


class Hand(Enum):
    """Which hand."""
    LEFT = "left"
    RIGHT = "right"


class Finger(Enum):
    """Finger identifiers."""
    THUMB = "thumb"
    INDEX = "index"
    MIDDLE = "middle"
    RING = "ring"
    PINKY = "pinky"


@dataclass
class HandState:
    """Position and finger state for one hand.

    Attributes
    ----------
    position : Point2D
        Current hand/mouse cursor position (pixels).
    fingers : dict[Finger, Point2D]
        Offset positions for each finger relative to the hand.
    on_device : str
        Device the hand is currently on (``"mouse"``, ``"keyboard"``,
        ``"touchscreen"``).
    """

    position: Point2D = field(default_factory=lambda: Point2D(0.0, 0.0))
    fingers: Dict[Finger, Point2D] = field(default_factory=dict)
    on_device: str = "mouse"


# ---------------------------------------------------------------------------
# Motor command
# ---------------------------------------------------------------------------


@dataclass
class MotorCommand:
    """A staged motor command.

    Motor actions in ACT-R proceed through three stages (Byrne &
    Anderson, 2001):
    1. Feature preparation (set movement parameters)
    2. Initiation (motor programme start)
    3. Execution (physical movement)

    Attributes
    ----------
    kind : str
        Command type (e.g. ``"punch"``, ``"move-cursor"``, ``"click"``).
    hand : Hand
        Which hand performs the action.
    target : Point2D | None
        Spatial target, if applicable.
    features : dict[str, Any]
        Additional parameters.
    preparation_time : float
        Computed preparation time (seconds).
    initiation_time : float
        Computed initiation time (seconds).
    execution_time : float
        Computed execution time (seconds).
    """

    kind: str
    hand: Hand = Hand.RIGHT
    target: Optional[Point2D] = None
    features: Dict[str, Any] = field(default_factory=dict)
    preparation_time: float = 0.0
    initiation_time: float = 0.0
    execution_time: float = 0.0

    @property
    def total_time(self) -> float:
        """Total motor command duration."""
        return self.preparation_time + self.initiation_time + self.execution_time


# ---------------------------------------------------------------------------
# ACT-R Motor Module
# ---------------------------------------------------------------------------


class ACTRMotorModule:
    """ACT-R motor module with staged execution and Fitts' law.

    The module tracks hand state and supports mouse movement, clicking,
    typing, and touch interactions.  Movement times use the Shannon
    formulation of Fitts' law (MacKenzie, 1992).

    Parameters
    ----------
    fitts_a : float
        Fitts' law intercept (seconds, default 0.050).
    fitts_b : float
        Fitts' law slope (seconds/bit, default 0.150).
    preparation_time : float
        Feature preparation time (seconds, default 0.050).
    initiation_time : float
        Motor initiation time (seconds, default 0.050).
    burst_time : float
        Minimum movement execution time (seconds, default 0.050).
    typing_rate : float
        Base inter-key interval for touch typing (seconds, default 0.040).
    homing_time : float
        Time to move hand between devices (seconds, default 0.400).
    """

    # Key-to-finger mapping for QWERTY (John, 1996)
    _KEY_FINGER: Dict[str, Tuple[Hand, Finger]] = {
        "q": (Hand.LEFT, Finger.PINKY), "w": (Hand.LEFT, Finger.RING),
        "e": (Hand.LEFT, Finger.MIDDLE), "r": (Hand.LEFT, Finger.INDEX),
        "t": (Hand.LEFT, Finger.INDEX),
        "a": (Hand.LEFT, Finger.PINKY), "s": (Hand.LEFT, Finger.RING),
        "d": (Hand.LEFT, Finger.MIDDLE), "f": (Hand.LEFT, Finger.INDEX),
        "g": (Hand.LEFT, Finger.INDEX),
        "z": (Hand.LEFT, Finger.PINKY), "x": (Hand.LEFT, Finger.RING),
        "c": (Hand.LEFT, Finger.MIDDLE), "v": (Hand.LEFT, Finger.INDEX),
        "b": (Hand.LEFT, Finger.INDEX),
        "y": (Hand.RIGHT, Finger.INDEX), "u": (Hand.RIGHT, Finger.INDEX),
        "i": (Hand.RIGHT, Finger.MIDDLE), "o": (Hand.RIGHT, Finger.RING),
        "p": (Hand.RIGHT, Finger.PINKY),
        "h": (Hand.RIGHT, Finger.INDEX), "j": (Hand.RIGHT, Finger.INDEX),
        "k": (Hand.RIGHT, Finger.MIDDLE), "l": (Hand.RIGHT, Finger.RING),
        ";": (Hand.RIGHT, Finger.PINKY),
        "n": (Hand.RIGHT, Finger.INDEX), "m": (Hand.RIGHT, Finger.INDEX),
        " ": (Hand.RIGHT, Finger.THUMB),
    }

    def __init__(
        self,
        fitts_a: float = 0.050,
        fitts_b: float = 0.150,
        preparation_time: float = 0.050,
        initiation_time: float = 0.050,
        burst_time: float = 0.050,
        typing_rate: float = 0.040,
        homing_time: float = 0.400,
    ) -> None:
        self.fitts_a = fitts_a
        self.fitts_b = fitts_b
        self.preparation_time = preparation_time
        self.initiation_time = initiation_time
        self.burst_time = burst_time
        self.typing_rate = typing_rate
        self.homing_time = homing_time

        self._left = HandState(
            position=Point2D(0.0, 0.0), on_device="keyboard"
        )
        self._right = HandState(
            position=Point2D(0.0, 0.0), on_device="mouse"
        )
        self._motor_buffer: Optional[MotorCommand] = None
        self._last_key: Optional[str] = None
        self._chunk_cache: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Hand state
    # ------------------------------------------------------------------ #

    def hand_state(self, hand: Hand) -> HandState:
        """Return the state of the specified hand."""
        return self._left if hand == Hand.LEFT else self._right

    @property
    def motor_buffer(self) -> Optional[MotorCommand]:
        """Currently staged motor command."""
        return self._motor_buffer

    def cursor_position(self) -> Point2D:
        """Current mouse cursor position (right hand)."""
        return self._right.position

    # ------------------------------------------------------------------ #
    # Fitts' law movement time
    # ------------------------------------------------------------------ #

    def fitts_time(self, distance: float, width: float) -> float:
        """Shannon-formulation Fitts' law movement time.

        .. math::

            MT = a + b \\cdot \\log_2\\!\\left(1 + \\frac{D}{W}\\right)

        Parameters
        ----------
        distance : float
            Movement distance (pixels).
        width : float
            Target width (pixels).

        Returns
        -------
        float
            Movement time in seconds.
        """
        distance = max(distance, 1.0)
        width = max(width, 1.0)
        return self.fitts_a + self.fitts_b * math.log2(1.0 + distance / width)

    def fitts_time_batch(
        self,
        distances: NDArray[np.floating],
        widths: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Vectorised Fitts' law over arrays."""
        d = np.maximum(np.asarray(distances, dtype=np.float64), 1.0)
        w = np.maximum(np.asarray(widths, dtype=np.float64), 1.0)
        return self.fitts_a + self.fitts_b * np.log2(1.0 + d / w)

    # ------------------------------------------------------------------ #
    # Mouse movement
    # ------------------------------------------------------------------ #

    def move_cursor(self, target: Point2D, width: float = 10.0) -> MotorCommand:
        """Plan a mouse cursor movement to *target*.

        Parameters
        ----------
        target : Point2D
            Destination.
        width : float
            Effective target width (pixels).

        Returns
        -------
        MotorCommand
            Staged motor command.
        """
        distance = self._right.position.distance_to(target)
        exec_time = self.fitts_time(distance, width)

        cmd = MotorCommand(
            kind="move-cursor",
            hand=Hand.RIGHT,
            target=target,
            preparation_time=self.preparation_time,
            initiation_time=self.initiation_time,
            execution_time=exec_time,
        )
        self._motor_buffer = cmd
        return cmd

    def execute_move(self, cmd: Optional[MotorCommand] = None) -> float:
        """Execute a move command, updating hand position.

        Returns
        -------
        float
            Total time taken (seconds).
        """
        cmd = cmd or self._motor_buffer
        if cmd is None or cmd.target is None:
            return 0.0

        self._right.position = cmd.target
        self._motor_buffer = None
        return cmd.total_time

    # ------------------------------------------------------------------ #
    # Click
    # ------------------------------------------------------------------ #

    def click(self, target: Optional[Point2D] = None, width: float = 10.0) -> float:
        """Move to *target* and click.

        Parameters
        ----------
        target : Point2D, optional
            Click destination.  If ``None``, clicks at current position.
        width : float
            Target width.

        Returns
        -------
        float
            Total time (move + click) in seconds.
        """
        total = 0.0
        if target is not None:
            cmd = self.move_cursor(target, width)
            total += self.execute_move(cmd)

        # Click: preparation + execution (no Fitts' component)
        click_time = self.preparation_time + self.burst_time
        total += click_time
        return total

    # ------------------------------------------------------------------ #
    # Typing model
    # ------------------------------------------------------------------ #

    def keystroke_time(
        self,
        key: str,
        skill_level: float = 1.0,
    ) -> float:
        """Predict time for a single keystroke.

        The inter-key interval depends on finger transition (same
        finger, same hand, or cross-hand) and typing skill (John, 1996).

        Parameters
        ----------
        key : str
            The character to type.
        skill_level : float
            Skill multiplier in (0, ∞).  1.0 = expert, > 1 = slower.

        Returns
        -------
        float
            Keystroke time in seconds.
        """
        base = self.typing_rate * skill_level

        # Transition cost
        transition = 0.0
        if self._last_key is not None:
            prev_mapping = self._KEY_FINGER.get(self._last_key.lower())
            curr_mapping = self._KEY_FINGER.get(key.lower())
            if prev_mapping and curr_mapping:
                if prev_mapping == curr_mapping:
                    transition = 0.020  # same finger penalty
                elif prev_mapping[0] == curr_mapping[0]:
                    transition = 0.010  # same hand penalty

        self._last_key = key
        return base + transition

    def typing_time(
        self,
        text: str,
        skill_level: float = 1.0,
    ) -> float:
        """Predict total time to type a string.

        Parameters
        ----------
        text : str
            Text to type.
        skill_level : float
            Typing skill multiplier.

        Returns
        -------
        float
            Total typing time in seconds.
        """
        if not text:
            return 0.0

        # Check if hand needs homing from mouse
        total = 0.0
        if self._right.on_device != "keyboard":
            total += self.homing_time
            self._right.on_device = "keyboard"

        for ch in text:
            total += self.keystroke_time(ch, skill_level)

        return total

    # ------------------------------------------------------------------ #
    # Touch interaction model
    # ------------------------------------------------------------------ #

    def touch_tap(
        self,
        target: Point2D,
        target_size: float = 44.0,
    ) -> float:
        """Predict time for a touch-screen tap.

        Uses an adapted Fitts' law with slightly higher intercept to
        account for the lower precision of finger input compared to
        mouse (Bi, Li & Zhai, 2013).

        Parameters
        ----------
        target : Point2D
            Tap destination.
        target_size : float
            Target diameter in pixels (default 44 px ≈ Apple HIG min).

        Returns
        -------
        float
            Tap time in seconds.
        """
        # Touch Fitts' parameters (higher than mouse)
        touch_a = 0.080
        touch_b = 0.180

        distance = self._right.position.distance_to(target)
        exec_time = touch_a + touch_b * math.log2(
            1.0 + distance / max(target_size, 1.0)
        )

        total = self.preparation_time + self.initiation_time + exec_time
        self._right.position = target
        return total

    def touch_swipe(
        self,
        start: Point2D,
        end: Point2D,
        precision: float = 50.0,
    ) -> float:
        """Predict time for a touch swipe gesture.

        Parameters
        ----------
        start : Point2D
            Swipe start point.
        end : Point2D
            Swipe end point.
        precision : float
            Effective width at swipe endpoint.

        Returns
        -------
        float
            Swipe time in seconds.
        """
        distance = start.distance_to(end)
        exec_time = self.fitts_time(distance, precision)
        # Swipe overhead (touch down + lift)
        overhead = 0.100
        total = self.preparation_time + exec_time + overhead
        self._right.position = end
        return total

    # ------------------------------------------------------------------ #
    # Device homing
    # ------------------------------------------------------------------ #

    def home_hand(self, hand: Hand, device: str) -> float:
        """Move a hand to a different input device.

        Parameters
        ----------
        hand : Hand
            Which hand to move.
        device : str
            Target device (``"mouse"``, ``"keyboard"``, ``"touchscreen"``).

        Returns
        -------
        float
            Homing time in seconds (0.0 if already on device).
        """
        state = self.hand_state(hand)
        if state.on_device == device:
            return 0.0
        state.on_device = device
        return self.homing_time

    # ------------------------------------------------------------------ #
    # Motor chunking for expert performance
    # ------------------------------------------------------------------ #

    def motor_chunk_time(
        self,
        sequence_key: str,
        base_time: float,
        n_executions: int,
        learning_rate: float = 0.3,
    ) -> float:
        """Predict time for a chunked (practiced) motor sequence.

        Repeated execution of the same motor sequence leads to chunking,
        where the sequence is executed as a single unit with reduced
        preparation overhead (power law of practice).

        .. math::

            T(n) = T_1 \\cdot n^{-\\alpha}

        Parameters
        ----------
        sequence_key : str
            Identifier for the motor sequence.
        base_time : float
            Time on the first execution *T₁* (seconds).
        n_executions : int
            Number of times this sequence has been executed.
        learning_rate : float
            Power law exponent α (default 0.3).

        Returns
        -------
        float
            Predicted sequence execution time.
        """
        n = max(1, n_executions)
        t = base_time * (n ** (-learning_rate))
        self._chunk_cache[sequence_key] = t
        return t

    # ------------------------------------------------------------------ #
    # Batch predictions
    # ------------------------------------------------------------------ #

    def click_sequence_time(
        self,
        targets: Sequence[Point2D],
        widths: Optional[Sequence[float]] = None,
    ) -> float:
        """Predict total time for a sequence of point-and-click actions.

        Parameters
        ----------
        targets : sequence of Point2D
            Click targets in order.
        widths : sequence of float, optional
            Target widths.  Defaults to 10.0 for each.

        Returns
        -------
        float
            Total interaction time in seconds.
        """
        if not targets:
            return 0.0

        ws = list(widths) if widths else [10.0] * len(targets)
        total = 0.0
        for target, w in zip(targets, ws):
            total += self.click(target, w)
        return total

    # ------------------------------------------------------------------ #
    # Movement preparation and execution decomposition
    # ------------------------------------------------------------------ #

    def decompose_movement(
        self,
        target: Point2D,
        width: float = 10.0,
    ) -> Dict[str, float]:
        """Decompose a movement into preparation, initiation, execution.

        Returns
        -------
        dict[str, float]
            Times for ``"preparation"``, ``"initiation"``, ``"execution"``,
            and ``"total"`` in seconds.
        """
        distance = self._right.position.distance_to(target)
        exec_t = self.fitts_time(distance, width)
        return {
            "preparation": self.preparation_time,
            "initiation": self.initiation_time,
            "execution": exec_t,
            "total": self.preparation_time + self.initiation_time + exec_t,
        }
