"""
Perceptual timing models for usability analysis.

This module implements empirically-grounded models of human visual perception
timing, based on established findings from eye-tracking and reading research.

Key references:
    - Rayner, K. (1998). Eye movements in reading and information processing:
      20 years of research. Psychological Bulletin, 124(3), 372-422.
    - Ware, C. (2012). Information Visualization: Perception for Design (3rd ed.).
      Morgan Kaufmann.
    - Dodge, R. (1900). Visual perception during eye movement.
      Psychological Review, 7(5), 454-465.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


class PerceptionModel:
    """Model of human visual perception timing.

    Provides empirically-calibrated estimates of the time required for various
    perceptual operations during interface use, including fixations, saccades,
    reading, icon recognition, and peripheral detection.

    All time values are in seconds unless otherwise noted.

    Attributes:
        MEAN_FIXATION_DURATION: Average fixation duration during reading and
            scene viewing. 250ms is the central estimate from Rayner (1998),
            Table 1, across multiple task types.
        MIN_SACCADE_DURATION: Minimum saccade duration for very short
            saccades (~1 degree amplitude). From Rayner (1998).
        MAX_SACCADE_DURATION: Maximum saccade duration for large saccades
            (~40 degrees amplitude). From Dodge (1900) and subsequent work.
        READING_RATE_WPM: Typical silent reading rate for college-age adults
            reading English prose. Rayner (1998), ~250 wpm for normal reading.
    """

    MEAN_FIXATION_DURATION: float = 0.250  # 250ms, Rayner (1998)
    MIN_SACCADE_DURATION: float = 0.020    # 20ms, minimum saccade
    MAX_SACCADE_DURATION: float = 0.200    # 200ms, maximum saccade
    READING_RATE_WPM: int = 250            # words per minute, Rayner (1998)

    # Foveal resolution constants (Ware, 2012)
    _FOVEAL_CHARS: int = 4       # characters clearly resolved at fovea
    _CHAR_FALLOFF: float = 0.5   # exponential decay rate with eccentricity

    def fixation_time(self, complexity: float = 1.0) -> float:
        """Estimate fixation duration modulated by visual complexity.

        More complex visual stimuli require longer fixations for adequate
        processing. Complexity acts as a linear multiplier on the mean
        fixation duration from Rayner (1998).

        Args:
            complexity: Multiplicative complexity factor. 1.0 represents
                normal reading text; values > 1.0 indicate more complex
                stimuli (e.g., dense diagrams, unfamiliar scripts).

        Returns:
            Fixation duration in seconds, clamped to [0.100, 0.600].

        References:
            Rayner, K. (1998). Eye movements in reading and information
            processing: 20 years of research. Psychological Bulletin,
            124(3), 372-422. Mean fixation ~250ms across tasks.
        """
        if complexity < 0:
            raise ValueError(f"complexity must be non-negative, got {complexity}")
        time = self.MEAN_FIXATION_DURATION * complexity
        return float(np.clip(time, 0.100, 0.600))

    def saccade_time(self, distance_degrees: float) -> float:
        """Estimate saccade duration from amplitude in visual degrees.

        Saccade duration increases approximately linearly with amplitude
        for amplitudes up to ~20 degrees, following the empirical formula:
            duration = 0.020 + 0.002 * amplitude_degrees

        This is consistent with the main sequence relationship reported
        in Rayner (1998) and Bahill et al. (1975).

        Args:
            distance_degrees: Saccade amplitude in degrees of visual angle.
                Must be non-negative.

        Returns:
            Saccade duration in seconds, clamped to
            [MIN_SACCADE_DURATION, MAX_SACCADE_DURATION].

        References:
            Rayner (1998); Bahill, Clark & Stark (1975). The main sequence:
            a tool for studying human eye movements.
        """
        if distance_degrees < 0:
            raise ValueError(
                f"distance_degrees must be non-negative, got {distance_degrees}"
            )
        time = 0.020 + 0.002 * distance_degrees
        return float(np.clip(time, self.MIN_SACCADE_DURATION, self.MAX_SACCADE_DURATION))

    def reading_time(self, n_words: int, word_complexity: float = 1.0) -> float:
        """Estimate reading time for a passage of text.

        Based on the typical silent reading rate of ~250 WPM for college-age
        adults reading English prose (Rayner, 1998). The word_complexity
        factor accounts for technical or unfamiliar vocabulary that slows
        reading.

        Args:
            n_words: Number of words in the passage. Must be non-negative.
            word_complexity: Multiplier for word difficulty. 1.0 = normal
                prose, >1.0 = technical/unfamiliar text.

        Returns:
            Estimated reading time in seconds.

        References:
            Rayner (1998), Section on reading rates: typical 250 WPM for
            normal reading, decreasing to ~200 WPM for technical material.
        """
        if n_words < 0:
            raise ValueError(f"n_words must be non-negative, got {n_words}")
        if word_complexity <= 0:
            raise ValueError(
                f"word_complexity must be positive, got {word_complexity}"
            )
        time = (n_words / self.READING_RATE_WPM) * 60.0 * word_complexity
        return float(time)

    def icon_recognition_time(self, familiarity: float = 0.5) -> float:
        """Estimate time to recognize an icon.

        Familiar icons (e.g., standard toolbar icons used daily) are recognized
        in ~200ms, while unfamiliar icons require ~600ms. Interpolates linearly
        based on familiarity.

        Based on findings from Ware (2012), Chapter 6, on preattentive
        processing and icon design.

        Args:
            familiarity: Familiarity with the icon, in [0.0, 1.0].
                0.0 = completely unfamiliar, 1.0 = highly familiar.

        Returns:
            Recognition time in seconds.

        References:
            Ware, C. (2012). Information Visualization: Perception for Design,
            3rd edition. Chapter 6: Visual Objects and Data Objects.
        """
        if not 0.0 <= familiarity <= 1.0:
            raise ValueError(
                f"familiarity must be in [0, 1], got {familiarity}"
            )
        time = 0.200 + 0.400 * (1.0 - familiarity)
        return float(time)

    def grouping_time(self, n_groups: int) -> float:
        """Estimate time for Gestalt perceptual grouping.

        The visual system groups elements by proximity, similarity, closure,
        and other Gestalt principles. Each additional group adds processing
        time. Based on Ware (2012), Chapter 5.

        Args:
            n_groups: Number of perceptual groups to process.
                Must be non-negative.

        Returns:
            Grouping perception time in seconds.

        References:
            Ware, C. (2012). Chapter 5: Visual Salience and Finding
            Information. Gestalt grouping adds ~50ms per group beyond
            a ~100ms base.
        """
        if n_groups < 0:
            raise ValueError(f"n_groups must be non-negative, got {n_groups}")
        time = 0.100 + 0.050 * n_groups
        return float(time)

    def color_discrimination_time(self, contrast: float = 1.0) -> float:
        """Estimate time to discriminate colors based on contrast.

        Higher contrast between foreground and background enables faster
        discrimination. The relationship is approximately inversely
        proportional to contrast ratio (Ware, 2012, Chapter 4).

        Args:
            contrast: Contrast ratio, must be > 0. A value of 1.0 represents
                a standard contrast level; higher values indicate greater
                contrast.

        Returns:
            Discrimination time in seconds, clamped to [0.100, 1.000].

        References:
            Ware, C. (2012). Chapter 4: Color. Weber-Fechner law applied
            to color discrimination timing.
        """
        if contrast <= 0:
            raise ValueError(f"contrast must be positive, got {contrast}")
        time = 0.150 / contrast
        return float(np.clip(time, 0.100, 1.000))

    @staticmethod
    def pixels_to_degrees(
        pixels: float,
        distance_mm: float = 600.0,
        ppi: float = 96.0,
    ) -> float:
        """Convert a pixel distance to degrees of visual angle.

        Uses the standard trigonometric formula for visual angle subtended
        by an object of known physical size at a known viewing distance:
            mm = pixels * 25.4 / ppi
            degrees = 2 * arctan(mm / (2 * distance_mm)) * (180 / pi)

        Args:
            pixels: Distance in pixels on screen.
            distance_mm: Viewing distance from eye to screen in millimeters.
                Default 600mm (~24 inches), a typical desktop viewing distance.
            ppi: Pixels per inch of the display. Default 96 (standard).

        Returns:
            Visual angle in degrees.

        References:
            Standard vision science formula. See Ware (2012), Appendix A
            for derivation.
        """
        if distance_mm <= 0:
            raise ValueError(f"distance_mm must be positive, got {distance_mm}")
        if ppi <= 0:
            raise ValueError(f"ppi must be positive, got {ppi}")
        mm = pixels * 25.4 / ppi
        degrees = 2.0 * math.atan(mm / (2.0 * distance_mm)) * (180.0 / math.pi)
        return float(degrees)

    @staticmethod
    def degrees_to_pixels(
        degrees: float,
        distance_mm: float = 600.0,
        ppi: float = 96.0,
    ) -> float:
        """Convert degrees of visual angle to pixel distance.

        Inverse of :meth:`pixels_to_degrees`.

        Args:
            degrees: Visual angle in degrees.
            distance_mm: Viewing distance in millimeters. Default 600mm.
            ppi: Pixels per inch of the display. Default 96.

        Returns:
            Distance in pixels.
        """
        if distance_mm <= 0:
            raise ValueError(f"distance_mm must be positive, got {distance_mm}")
        if ppi <= 0:
            raise ValueError(f"ppi must be positive, got {ppi}")
        radians = degrees * (math.pi / 180.0)
        mm = 2.0 * distance_mm * math.tan(radians / 2.0)
        pixels = mm * ppi / 25.4
        return float(pixels)

    def visual_span(self, eccentricity_degrees: float) -> int:
        """Estimate the number of characters recognizable at a given eccentricity.

        At the fovea (0 degrees eccentricity), approximately 4 characters can
        be resolved simultaneously. Recognition drops off with increasing
        eccentricity following an approximately exponential decay.

        This models the "perceptual span" concept from Rayner (1998), where
        useful information is extracted from a limited region around fixation.

        Args:
            eccentricity_degrees: Distance from foveal center in degrees
                of visual angle. Must be non-negative.

        Returns:
            Estimated number of recognizable characters (integer, >= 0).

        References:
            Rayner (1998): perceptual span of 3-4 characters to the left
            and 14-15 to the right of fixation during English reading,
            but letter identification limited to ~4 characters around
            fixation.
        """
        if eccentricity_degrees < 0:
            raise ValueError(
                f"eccentricity_degrees must be non-negative, got {eccentricity_degrees}"
            )
        chars = self._FOVEAL_CHARS * math.exp(
            -self._CHAR_FALLOFF * eccentricity_degrees
        )
        return max(0, int(round(chars)))

    def text_legibility_factor(
        self,
        font_size_pt: float,
        viewing_distance_mm: float = 600.0,
    ) -> float:
        """Compute a legibility factor (0-1) based on angular size of text.

        Larger angular size (bigger font or closer distance) yields higher
        legibility. The factor approaches 1.0 when the text subtends
        a comfortable reading angle (~0.3 degrees, roughly 12pt at 600mm)
        and falls toward 0 for very small angular sizes.

        Uses a logistic sigmoid centered on the comfortable reading threshold.

        Args:
            font_size_pt: Font size in typographic points (1pt = 1/72 inch).
            viewing_distance_mm: Distance from eye to screen in mm.

        Returns:
            Legibility factor in [0.0, 1.0]. Values near 1.0 indicate
            highly legible text; values near 0.0 indicate barely legible.

        References:
            Ware, C. (2012). Chapter 3: Lightness, Brightness, Contrast,
            and Constancy. Discusses minimum angular size for legibility.
        """
        if font_size_pt <= 0:
            raise ValueError(f"font_size_pt must be positive, got {font_size_pt}")
        if viewing_distance_mm <= 0:
            raise ValueError(
                f"viewing_distance_mm must be positive, got {viewing_distance_mm}"
            )
        # Convert point size to mm (1pt = 0.3528mm)
        font_mm = font_size_pt * 0.3528
        angular_size_deg = (
            2.0 * math.atan(font_mm / (2.0 * viewing_distance_mm))
            * (180.0 / math.pi)
        )
        # Comfortable reading threshold ~0.3 degrees (~12pt at 600mm)
        comfortable_threshold = 0.30
        # Logistic sigmoid: steep transition around threshold
        steepness = 20.0
        factor = 1.0 / (
            1.0 + math.exp(-steepness * (angular_size_deg - comfortable_threshold * 0.5))
        )
        return float(np.clip(factor, 0.0, 1.0))

    def peripheral_detection_probability(
        self,
        eccentricity_degrees: float,
        target_size_degrees: float,
    ) -> float:
        """Estimate probability of detecting a target in peripheral vision.

        Detection probability decreases with eccentricity and increases with
        target size, following the cortical magnification factor. The model
        combines eccentricity decay with a size-dependent detection boost.

        Args:
            eccentricity_degrees: Distance from foveal center in degrees.
                Must be non-negative.
            target_size_degrees: Angular size of the target in degrees.
                Must be positive.

        Returns:
            Detection probability in [0.0, 1.0].

        References:
            Ware, C. (2012). Chapter 2: The Environment, Optics, Resolution,
            and the Visual System. Cortical magnification and peripheral
            acuity.
            Rayner (1998): useful field of view and parafoveal processing.
        """
        if eccentricity_degrees < 0:
            raise ValueError(
                f"eccentricity_degrees must be non-negative, got {eccentricity_degrees}"
            )
        if target_size_degrees <= 0:
            raise ValueError(
                f"target_size_degrees must be positive, got {target_size_degrees}"
            )
        # Eccentricity decay: cortical magnification factor (Ware, 2012)
        # Acuity drops roughly as 1 / (1 + 0.33 * eccentricity)
        eccentricity_factor = 1.0 / (1.0 + 0.33 * eccentricity_degrees)

        # Size boost: larger targets are easier to detect peripherally.
        # Saturates via sigmoid — targets > ~2 degrees easily detected.
        size_factor = 1.0 - math.exp(-2.0 * target_size_degrees)

        probability = eccentricity_factor * size_factor
        return float(np.clip(probability, 0.0, 1.0))
