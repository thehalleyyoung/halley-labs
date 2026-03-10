"""Unit tests for usability_oracle.cognitive.motor.MotorModel.

Tests cover keystroke timing at different skill levels, click and double-click
timing, drag operations, scroll timing, homing, mental preparation, system
response, gesture timing, typing prediction, KLM sequence execution, and
KLM constant values.

References
----------
Card, S. K., Moran, T. P., & Newell, A. (1983). *The Psychology of
    Human-Computer Interaction*. Lawrence Erlbaum Associates.
"""

from __future__ import annotations

import pytest

from usability_oracle.cognitive.motor import MotorModel


# ------------------------------------------------------------------ #
# KLM constants
# ------------------------------------------------------------------ #


class TestKLMConstants:
    """Verify the published KLM timing constants (Card et al., 1983)."""

    def test_k_expert(self) -> None:
        """K_EXPERT should be 0.120 s (135 wpm)."""
        assert MotorModel.K_EXPERT == pytest.approx(0.120)

    def test_k_good(self) -> None:
        """K_GOOD should be 0.200 s (90 wpm)."""
        assert MotorModel.K_GOOD == pytest.approx(0.200)

    def test_k_average(self) -> None:
        """K_AVERAGE should be 0.280 s (55 wpm)."""
        assert MotorModel.K_AVERAGE == pytest.approx(0.280)

    def test_k_poor(self) -> None:
        """K_POOR should be 0.500 s (hunt-and-peck)."""
        assert MotorModel.K_POOR == pytest.approx(0.500)

    def test_p_pointing(self) -> None:
        """P_POINTING should be 1.100 s."""
        assert MotorModel.P_POINTING == pytest.approx(1.100)

    def test_h_homing(self) -> None:
        """H_HOMING should be 0.400 s."""
        assert MotorModel.H_HOMING == pytest.approx(0.400)

    def test_m_mental(self) -> None:
        """M_MENTAL should be 1.350 s."""
        assert MotorModel.M_MENTAL == pytest.approx(1.350)


# ------------------------------------------------------------------ #
# Keystroke time
# ------------------------------------------------------------------ #


class TestKeystrokeTime:
    """Tests for MotorModel.keystroke_time()."""

    def test_regular_average(self) -> None:
        """Regular key, average typist → 0.280 s."""
        result = MotorModel.keystroke_time("regular", "average")
        assert result == pytest.approx(0.280)

    def test_regular_expert(self) -> None:
        """Regular key, expert typist → 0.120 s."""
        result = MotorModel.keystroke_time("regular", "expert")
        assert result == pytest.approx(0.120)

    def test_regular_poor(self) -> None:
        """Regular key, poor typist → 0.500 s."""
        result = MotorModel.keystroke_time("regular", "poor")
        assert result == pytest.approx(0.500)

    def test_shift_key_multiplier(self) -> None:
        """Shift key uses 1.25× multiplier.

        Average: 0.280 * 1.25 = 0.350 s.
        """
        result = MotorModel.keystroke_time("shift_key", "average")
        assert result == pytest.approx(0.280 * 1.25)

    def test_function_key_multiplier(self) -> None:
        """Function key uses 1.10× multiplier.

        Average: 0.280 * 1.10 = 0.308 s.
        """
        result = MotorModel.keystroke_time("function_key", "average")
        assert result == pytest.approx(0.280 * 1.10)

    def test_modifier_combo_multiplier(self) -> None:
        """Modifier combo (e.g. Ctrl+Shift+K) uses 1.50× multiplier.

        Average: 0.280 * 1.50 = 0.420 s.
        """
        result = MotorModel.keystroke_time("modifier_combo", "average")
        assert result == pytest.approx(0.280 * 1.50)

    def test_expert_faster_than_poor(self) -> None:
        """Expert keystroke time should be less than poor typist time."""
        expert = MotorModel.keystroke_time("regular", "expert")
        poor = MotorModel.keystroke_time("regular", "poor")
        assert expert < poor


# ------------------------------------------------------------------ #
# Click and double-click
# ------------------------------------------------------------------ #


class TestClickTiming:
    """Tests for click_time() and double_click_time()."""

    def test_click_time(self) -> None:
        """Single click should take 0.200 s."""
        assert MotorModel.click_time() == pytest.approx(0.200)

    def test_double_click_time(self) -> None:
        """Double click should take 0.400 s."""
        assert MotorModel.double_click_time() == pytest.approx(0.400)

    def test_double_click_longer(self) -> None:
        """Double click must take longer than single click."""
        assert MotorModel.double_click_time() > MotorModel.click_time()


# ------------------------------------------------------------------ #
# Drag time
# ------------------------------------------------------------------ #


class TestDragTime:
    """Tests for MotorModel.drag_time()."""

    def test_drag_positive(self) -> None:
        """Drag time should be positive for any distance and precision."""
        result = MotorModel.drag_time(distance=200, precision=20)
        assert result > 0.0

    def test_drag_increases_with_distance(self) -> None:
        """Longer drag distance → more time (via Fitts' law component)."""
        short = MotorModel.drag_time(distance=50, precision=20)
        long = MotorModel.drag_time(distance=500, precision=20)
        assert long > short

    def test_drag_increases_with_precision(self) -> None:
        """Higher precision (smaller width) → more time.

        precision is the effective target width; smaller = harder.
        """
        easy = MotorModel.drag_time(distance=200, precision=50)
        hard = MotorModel.drag_time(distance=200, precision=5)
        assert hard > easy

    def test_drag_includes_overhead(self) -> None:
        """Drag time should include button-down/up overhead of 0.1 s."""
        result = MotorModel.drag_time(distance=100, precision=50)
        fitts = MotorModel._fitts_time(100, 50)
        assert result == pytest.approx(0.100 + fitts, rel=1e-6)


# ------------------------------------------------------------------ #
# Scroll time
# ------------------------------------------------------------------ #


class TestScrollTime:
    """Tests for MotorModel.scroll_time()."""

    def test_scroll_basic(self) -> None:
        """Scroll 1000 px at 1000 px/s → 1.0 s + 0.2 s adjustment = 1.2 s."""
        result = MotorModel.scroll_time(1000)
        assert result == pytest.approx(1.2, rel=1e-6)

    def test_scroll_zero_distance(self) -> None:
        """Zero scroll distance → only adjustment time (0.2 s)."""
        result = MotorModel.scroll_time(0)
        assert result == pytest.approx(0.2, rel=1e-6)

    def test_scroll_positive(self) -> None:
        """Scroll time should always be positive."""
        assert MotorModel.scroll_time(500) > 0.0


# ------------------------------------------------------------------ #
# Homing, mental preparation, system response
# ------------------------------------------------------------------ #


class TestMiscOperators:
    """Tests for homing, mental preparation, and system response times."""

    def test_homing_time(self) -> None:
        """Homing time should be H_HOMING = 0.400 s."""
        assert MotorModel.homing_time() == pytest.approx(0.400)

    def test_mental_preparation_time(self) -> None:
        """Mental preparation should be M_MENTAL = 1.350 s."""
        assert MotorModel.mental_preparation_time() == pytest.approx(1.350)

    def test_system_response_default(self) -> None:
        """Default system response should be 0.100 s."""
        assert MotorModel.system_response_time() == pytest.approx(0.100)

    def test_system_response_custom(self) -> None:
        """Custom system response value should be returned as-is."""
        assert MotorModel.system_response_time(0.5) == pytest.approx(0.5)

    def test_system_response_non_negative(self) -> None:
        """Negative input should be clamped to 0."""
        assert MotorModel.system_response_time(-1.0) == pytest.approx(0.0)


# ------------------------------------------------------------------ #
# Gesture time
# ------------------------------------------------------------------ #


class TestGestureTime:
    """Tests for MotorModel.gesture_time()."""

    def test_simple_gesture(self) -> None:
        """Single segment, complexity=1 → 0.150 * 1 * 1 + 0 = 0.150 s."""
        result = MotorModel.gesture_time(complexity=1.0, n_segments=1)
        assert result == pytest.approx(0.150)

    def test_multi_segment(self) -> None:
        """3 segments, complexity=1 → 0.150 * 3 + 0.050 * 2 = 0.550 s."""
        result = MotorModel.gesture_time(complexity=1.0, n_segments=3)
        expected = 0.150 * 3 + 0.050 * 2
        assert result == pytest.approx(expected, rel=1e-6)

    def test_complex_gesture(self) -> None:
        """Higher complexity → longer gesture time."""
        simple = MotorModel.gesture_time(complexity=1.0, n_segments=2)
        complex_g = MotorModel.gesture_time(complexity=3.0, n_segments=2)
        assert complex_g > simple

    def test_more_segments_longer(self) -> None:
        """More segments → longer gesture time."""
        short = MotorModel.gesture_time(complexity=1.5, n_segments=2)
        long = MotorModel.gesture_time(complexity=1.5, n_segments=5)
        assert long > short


# ------------------------------------------------------------------ #
# Typing time
# ------------------------------------------------------------------ #


class TestTypingTime:
    """Tests for MotorModel.typing_time()."""

    def test_empty_string(self) -> None:
        """Empty string → 0.0 s."""
        assert MotorModel.typing_time("") == pytest.approx(0.0)

    def test_lowercase_only(self) -> None:
        """All lowercase → all regular keystrokes.

        'abc' → 3 * 0.280 = 0.840 s (average typist).
        """
        result = MotorModel.typing_time("abc", skill_level="average")
        expected = 3 * MotorModel.K_AVERAGE
        assert result == pytest.approx(expected, rel=1e-6)

    def test_uppercase_uses_shift(self) -> None:
        """Uppercase letters should use shift_key multiplier.

        'AB' → 2 * 0.280 * 1.25 = 0.700 s.
        """
        result = MotorModel.typing_time("AB", skill_level="average")
        expected = 2 * MotorModel.K_AVERAGE * 1.25
        assert result == pytest.approx(expected, rel=1e-6)

    def test_expert_faster_typing(self) -> None:
        """Expert typist should be faster than average for same text."""
        expert = MotorModel.typing_time("hello world", "expert")
        average = MotorModel.typing_time("hello world", "average")
        assert expert < average


# ------------------------------------------------------------------ #
# KLM sequence
# ------------------------------------------------------------------ #


class TestKLMSequence:
    """Tests for MotorModel.klm_sequence()."""

    def test_single_keystroke(self) -> None:
        """Sequence ['K'] → one regular keystroke time."""
        result = MotorModel.klm_sequence(["K"], skill_level="average")
        assert result == pytest.approx(MotorModel.K_AVERAGE, rel=1e-6)

    def test_typical_sequence(self) -> None:
        """M + P + C (mental, point, click) → 1.350 + 1.100 + 0.200 = 2.650 s."""
        result = MotorModel.klm_sequence(["M", "P", "C"])
        expected = 1.350 + 1.100 + 0.200
        assert result == pytest.approx(expected, rel=1e-6)

    def test_all_operators(self) -> None:
        """Verify all valid operator codes are accepted without error."""
        ops = ["K", "Ks", "Kf", "Km", "P", "H", "M", "R", "C", "D"]
        result = MotorModel.klm_sequence(ops)
        assert result > 0.0

    def test_invalid_operator_raises(self) -> None:
        """Unknown operator code should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown KLM operator"):
            MotorModel.klm_sequence(["X"])

    def test_homing_operator(self) -> None:
        """H operator should add H_HOMING time."""
        result = MotorModel.klm_sequence(["H"])
        assert result == pytest.approx(MotorModel.H_HOMING, rel=1e-6)

    def test_system_response_operator(self) -> None:
        """R operator should use the system_response parameter."""
        result = MotorModel.klm_sequence(["R"], system_response=0.5)
        assert result == pytest.approx(0.5, rel=1e-6)

    def test_empty_sequence(self) -> None:
        """Empty operator sequence → 0.0 s."""
        assert MotorModel.klm_sequence([]) == pytest.approx(0.0)

    def test_double_click_operator(self) -> None:
        """D operator should add double_click_time."""
        result = MotorModel.klm_sequence(["D"])
        assert result == pytest.approx(0.400, rel=1e-6)
