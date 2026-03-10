"""
Published cognitive parameter ranges for usability modelling.

Provides empirically-sourced parameter sets for Fitts' law, Hick-Hyman law,
visual search, working memory, and motor timing (KLM). Parameters are given
as ranges with typical values and literature sources, supporting both
point-estimate and interval-arithmetic analyses.

Key references:
    - Fitts, P.M. (1954). The information capacity of the human motor system
      in controlling the amplitude of movement. J. Exp. Psych., 47(6), 381-391.
    - Hick, W.E. (1952). On the rate of gain of information. Q. J. Exp. Psych.,
      4(1), 11-26.
    - Card, S.K., Moran, T.P., & Newell, A. (1983). The Psychology of
      Human-Computer Interaction. Lawrence Erlbaum Associates.
    - Wolfe, J.M. (1998). Visual Search. In H. Pashler (Ed.), Attention.
      Psychology Press.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    from usability_oracle.interval.interval import Interval
except ImportError:
    # Graceful fallback: define a minimal Interval for standalone use
    class Interval:
        """Minimal interval placeholder when full module is unavailable."""

        def __init__(self, lo: float, hi: float) -> None:
            self.lo = lo
            self.hi = hi

        def __repr__(self) -> str:
            return f"Interval({self.lo}, {self.hi})"


class CognitiveParameters:
    """Published cognitive parameter ranges for human performance modelling.

    Each parameter set is a dictionary containing 'min', 'max', 'typical',
    and 'source' keys. These ranges reflect inter-individual variability
    reported across multiple empirical studies.

    Class-level constants hold the canonical parameter dictionaries.
    Instance methods support percentile-based parameter selection and
    interval construction for uncertainty-aware analyses.
    """

    # ------------------------------------------------------------------
    # Fitts' Law:  MT = a + b * log2(2D / W)
    # ------------------------------------------------------------------
    FITTS_PARAMETERS: Dict[str, Dict[str, Any]] = {
        "a": {
            "min": 0.000,
            "max": 0.100,
            "typical": 0.050,
            "source": (
                "Fitts (1954); MacKenzie (1992). Intercept term varies with "
                "device. 0ms for well-practised mouse, up to 100ms for "
                "touchscreen or stylus."
            ),
        },
        "b": {
            "min": 0.100,
            "max": 0.250,
            "typical": 0.150,
            "source": (
                "Fitts (1954); Soukoreff & MacKenzie (2004). Slope term "
                "(seconds per bit) ranges from ~100ms/bit for mouse to "
                "~250ms/bit for head-pointer devices."
            ),
        },
    }

    # ------------------------------------------------------------------
    # Hick-Hyman Law:  RT = a + b * log2(n)
    # ------------------------------------------------------------------
    HICK_PARAMETERS: Dict[str, Dict[str, Any]] = {
        "a": {
            "min": 0.150,
            "max": 0.300,
            "typical": 0.200,
            "source": (
                "Hick (1952); Hyman (1953). Base reaction time component, "
                "typically 150-300ms depending on stimulus-response "
                "compatibility."
            ),
        },
        "b": {
            "min": 0.100,
            "max": 0.200,
            "typical": 0.155,
            "source": (
                "Hick (1952); Hyman (1953). Information processing rate "
                "~155ms per bit of entropy for choice reactions."
            ),
        },
    }

    # ------------------------------------------------------------------
    # Visual Search Slopes (ms per item)
    # ------------------------------------------------------------------
    VISUAL_SEARCH_SLOPES: Dict[str, Dict[str, Any]] = {
        "efficient": {
            "min": 0.002,
            "max": 0.006,
            "typical": 0.004,
            "source": (
                "Wolfe (1998); Treisman & Gelade (1980). Feature search "
                "with pop-out: 2-6ms per item, near-parallel processing."
            ),
        },
        "inefficient": {
            "min": 0.020,
            "max": 0.030,
            "typical": 0.025,
            "source": (
                "Wolfe (1998). Serial self-terminating search: 20-30ms "
                "per item for target-absent or heterogeneous displays."
            ),
        },
        "conjunction": {
            "min": 0.010,
            "max": 0.020,
            "typical": 0.015,
            "source": (
                "Treisman & Gelade (1980); Wolfe (1998). Conjunction "
                "search: 10-20ms per item, guided serial search."
            ),
        },
    }

    # ------------------------------------------------------------------
    # Working Memory Parameters
    # ------------------------------------------------------------------
    WORKING_MEMORY_PARAMS: Dict[str, Dict[str, Any]] = {
        "capacity": {
            "min": 3,
            "max": 7,
            "typical": 4,
            "source": (
                "Cowan (2001). The magical number 4 in short-term memory. "
                "Range 3-7 reflects individual differences; 4 +/- 1 is "
                "the modern estimate replacing Miller's 7 +/- 2."
            ),
        },
        "decay_half_life": {
            "min": 5.0,
            "max": 15.0,
            "typical": 9.0,
            "source": (
                "Barrouillet et al. (2004). Time and cognitive load in "
                "working memory. Decay half-life ~9s without rehearsal."
            ),
        },
    }

    # ------------------------------------------------------------------
    # KLM Motor Parameters (Card, Moran & Newell, 1983)
    # ------------------------------------------------------------------
    MOTOR_PARAMS: Dict[str, Dict[str, Any]] = {
        "K": {
            "min": 0.080,
            "max": 0.280,
            "typical": 0.200,
            "source": (
                "Card, Moran & Newell (1983). Keystroke time: best typist "
                "~80ms, average ~200ms, worst ~280ms."
            ),
        },
        "P": {
            "min": 0.800,
            "max": 1.500,
            "typical": 1.100,
            "source": (
                "Card, Moran & Newell (1983). Pointing time: varies by "
                "device and distance/width, average ~1.1s."
            ),
        },
        "H": {
            "min": 0.300,
            "max": 0.500,
            "typical": 0.400,
            "source": (
                "Card, Moran & Newell (1983). Homing — time to move hand "
                "between keyboard and mouse, ~400ms."
            ),
        },
        "M": {
            "min": 0.600,
            "max": 1.500,
            "typical": 1.200,
            "source": (
                "Card, Moran & Newell (1983). Mental preparation time, "
                "~1.2s average. Highly variable with task."
            ),
        },
        "R": {
            "min": 0.050,
            "max": 2.000,
            "typical": 0.100,
            "source": (
                "Card, Moran & Newell (1983). System response time. "
                "Highly variable; 100ms is a responsive system."
            ),
        },
    }

    # ------------------------------------------------------------------
    # Perceptual Parameters
    # ------------------------------------------------------------------
    PERCEPTUAL_PARAMS: Dict[str, Dict[str, Any]] = {
        "fixation_duration": {
            "min": 0.150,
            "max": 0.500,
            "typical": 0.250,
            "source": "Rayner (1998). Fixation duration 150-500ms, mean ~250ms.",
        },
        "saccade_duration": {
            "min": 0.020,
            "max": 0.200,
            "typical": 0.040,
            "source": (
                "Rayner (1998). Saccade duration 20-200ms depending on "
                "amplitude; typical reading saccade ~40ms."
            ),
        },
        "reading_rate_wpm": {
            "min": 150,
            "max": 400,
            "typical": 250,
            "source": (
                "Rayner (1998). Silent reading rate 150-400 WPM, "
                "typical college student ~250 WPM."
            ),
        },
    }

    # ------------------------------------------------------------------
    # Population Percentile Ranges
    # ------------------------------------------------------------------
    POPULATION_RANGES: Dict[int, Dict[str, Any]] = {
        5: {
            "label": "young_expert",
            "fitts_b": 0.100,
            "hick_b": 0.100,
            "keystroke": 0.080,
            "mental_prep": 0.600,
            "reading_wpm": 400,
            "wm_capacity": 7,
            "description": "5th percentile: young expert user, fastest responses.",
        },
        25: {
            "label": "above_average",
            "fitts_b": 0.120,
            "hick_b": 0.130,
            "keystroke": 0.120,
            "mental_prep": 0.800,
            "reading_wpm": 325,
            "wm_capacity": 5,
            "description": "25th percentile: above-average user.",
        },
        50: {
            "label": "typical",
            "fitts_b": 0.150,
            "hick_b": 0.155,
            "keystroke": 0.200,
            "mental_prep": 1.200,
            "reading_wpm": 250,
            "wm_capacity": 4,
            "description": "50th percentile: typical adult user.",
        },
        75: {
            "label": "below_average",
            "fitts_b": 0.190,
            "hick_b": 0.175,
            "keystroke": 0.240,
            "mental_prep": 1.400,
            "reading_wpm": 200,
            "wm_capacity": 3,
            "description": "75th percentile: below-average or less practiced.",
        },
        95: {
            "label": "elderly_novice",
            "fitts_b": 0.250,
            "hick_b": 0.200,
            "keystroke": 0.280,
            "mental_prep": 1.500,
            "reading_wpm": 150,
            "wm_capacity": 3,
            "description": (
                "95th percentile: elderly or novice user, slowest responses."
            ),
        },
    }

    # Sorted percentile keys for interpolation
    _PERCENTILE_KEYS = sorted(POPULATION_RANGES.keys())

    def get_parameter_set(self, percentile: float = 50) -> Dict[str, Any]:
        """Return an interpolated parameter set for the given percentile.

        Linearly interpolates between the two nearest published percentile
        ranges. For example, percentile=37 interpolates between the 25th
        and 50th percentile entries.

        Args:
            percentile: Population percentile in [0, 100].

        Returns:
            Dictionary of cognitive parameters at the requested percentile.
        """
        if not 0 <= percentile <= 100:
            raise ValueError(f"percentile must be in [0, 100], got {percentile}")

        keys = self._PERCENTILE_KEYS

        # Clamp to available range
        if percentile <= keys[0]:
            return copy.deepcopy(self.POPULATION_RANGES[keys[0]])
        if percentile >= keys[-1]:
            return copy.deepcopy(self.POPULATION_RANGES[keys[-1]])

        # Find bracketing percentiles
        lower_key = keys[0]
        upper_key = keys[-1]
        for i in range(len(keys) - 1):
            if keys[i] <= percentile <= keys[i + 1]:
                lower_key = keys[i]
                upper_key = keys[i + 1]
                break

        lower = self.POPULATION_RANGES[lower_key]
        upper = self.POPULATION_RANGES[upper_key]

        # Interpolation fraction
        span = upper_key - lower_key
        frac = (percentile - lower_key) / span if span > 0 else 0.0

        result: Dict[str, Any] = {
            "label": f"percentile_{int(percentile)}",
            "description": f"Interpolated parameters at {percentile}th percentile.",
        }
        numeric_keys = [
            "fitts_b", "hick_b", "keystroke", "mental_prep",
            "reading_wpm", "wm_capacity",
        ]
        for k in numeric_keys:
            lo_val = float(lower[k])
            hi_val = float(upper[k])
            result[k] = lo_val + frac * (hi_val - lo_val)

        # Round integer-like parameters
        result["wm_capacity"] = int(round(result["wm_capacity"]))
        result["reading_wpm"] = int(round(result["reading_wpm"]))

        return result

    def get_conservative_parameters(self) -> Dict[str, Any]:
        """Return parameters for a slow / worst-case user (95th percentile).

        Use these parameters for conservative usability estimates that
        accommodate elderly or novice users.

        Returns:
            Parameter set at the 95th percentile.
        """
        return self.get_parameter_set(percentile=95)

    def get_typical_parameters(self) -> Dict[str, Any]:
        """Return parameters for a typical user (50th percentile).

        Returns:
            Parameter set at the 50th percentile.
        """
        return self.get_parameter_set(percentile=50)

    def get_optimistic_parameters(self) -> Dict[str, Any]:
        """Return parameters for a fast / best-case user (5th percentile).

        Use these parameters for optimistic estimates reflecting young,
        expert users.

        Returns:
            Parameter set at the 5th percentile.
        """
        return self.get_parameter_set(percentile=5)

    def as_intervals(
        self,
        percentile_low: float = 25,
        percentile_high: float = 75,
    ) -> Dict[str, "Interval"]:
        """Return parameter ranges as Interval objects.

        Constructs an Interval [lo, hi] for each numeric parameter by
        evaluating at the low and high percentiles.

        Args:
            percentile_low: Lower bound percentile (default 25).
            percentile_high: Upper bound percentile (default 75).

        Returns:
            Dictionary mapping parameter names to Interval instances.
        """
        if percentile_low >= percentile_high:
            raise ValueError(
                f"percentile_low ({percentile_low}) must be less than "
                f"percentile_high ({percentile_high})"
            )
        lo_params = self.get_parameter_set(percentile_low)
        hi_params = self.get_parameter_set(percentile_high)

        numeric_keys = [
            "fitts_b", "hick_b", "keystroke", "mental_prep",
            "reading_wpm", "wm_capacity",
        ]
        intervals: Dict[str, Interval] = {}
        for k in numeric_keys:
            lo_val = float(lo_params[k])
            hi_val = float(hi_params[k])
            # Ensure lo <= hi (some params decrease with percentile, e.g., WPM)
            if lo_val > hi_val:
                lo_val, hi_val = hi_val, lo_val
            intervals[k] = Interval(lo_val, hi_val)

        return intervals
