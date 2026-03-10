"""
usability_oracle.core.constants — Physical and cognitive constants.

Every constant includes a literature citation in its docstring.  Where a
parameter has a known population range, we store it as an
:class:`~usability_oracle.core.types.Interval` so that downstream
interval-arithmetic propagates uncertainty automatically.

Units are **seconds** for times, **millimetres** for distances (unless
noted), and **bits/second** for information throughput.
"""

from __future__ import annotations

from typing import Any, Dict

from usability_oracle.core.types import Interval

# ═══════════════════════════════════════════════════════════════════════════
# Fitts' Law coefficients
# ═══════════════════════════════════════════════════════════════════════════

FITTS_A_RANGE: Interval = Interval(0.020, 0.100)
"""Fitts' Law intercept *a* (seconds).

Population range across standard pointing devices (mouse, trackpad,
touchscreen).  The lower bound corresponds to expert mouse users; the
upper bound to touchpad / low-dexterity populations.

Citation: MacKenzie, I. S. (1992). Fitts' law as a research and design
tool in human-computer interaction. *Human-Computer Interaction*, 7(1),
91-139.
"""

FITTS_B_RANGE: Interval = Interval(0.100, 0.250)
"""Fitts' Law slope *b* (seconds / bit).

Represents the reciprocal of the index of performance (IP).
IP typically ranges 3.7-10.4 bits/s across devices.

Citation: MacKenzie, I. S. (1992).
"""

# ═══════════════════════════════════════════════════════════════════════════
# Hick-Hyman Law coefficients
# ═══════════════════════════════════════════════════════════════════════════

HICK_A_RANGE: Interval = Interval(0.150, 0.300)
"""Hick-Hyman intercept *a* (seconds).

Represents the base reaction time independent of the number of choices.

Citation: Hick, W. E. (1952). On the rate of gain of information.
*Quarterly Journal of Experimental Psychology*, 4(1), 11-26.
"""

HICK_B_RANGE: Interval = Interval(0.100, 0.200)
"""Hick-Hyman slope *b* (seconds / bit).

Each additional bit of choice entropy adds *b* seconds to the reaction
time: RT = a + b * log2(n + 1).

Citation: Hyman, R. (1953). Stimulus information as a determinant of
reaction time. *Journal of Experimental Psychology*, 45(3), 188-196.
"""

# ═══════════════════════════════════════════════════════════════════════════
# Working Memory
# ═══════════════════════════════════════════════════════════════════════════

WORKING_MEMORY_CAPACITY: Interval = Interval(3.0, 5.0)
"""Effective working-memory capacity (chunks).

Miller (1956) proposed 7 +/- 2; Cowan (2001) revised this to ~4 for
unrelated items.  We use [3, 5] to bracket the population.

Citation: Cowan, N. (2001). The magical number 4 in short-term memory.
*Behavioral and Brain Sciences*, 24(1), 87-114.
"""

WORKING_MEMORY_DECAY_HALF_LIFE: Interval = Interval(7.0, 15.0)
"""Working-memory decay half-life (seconds).

Time for recall probability to drop to 50%.  Ranges from ~7 s under
high interference to ~15 s with rehearsal.

Citation: Barrouillet, P. et al. (2004). Time constraints and resource
sharing in adults' working memory spans. *Journal of Experimental
Psychology: General*, 133(1), 83-100.
"""

# ═══════════════════════════════════════════════════════════════════════════
# Visual search
# ═══════════════════════════════════════════════════════════════════════════

VISUAL_SEARCH_EFFICIENT_SLOPE: Interval = Interval(0.005, 0.015)
"""Visual search slope for *efficient* (pop-out) search (seconds / item).

Feature-present searches with a unique conjunction typically yield slopes
under 10 ms/item.

Citation: Wolfe, J. M. (1998). Visual search. In H. Pashler (Ed.),
*Attention* (pp. 13-73). Psychology Press.
"""

VISUAL_SEARCH_INEFFICIENT_SLOPE: Interval = Interval(0.020, 0.050)
"""Visual search slope for *inefficient* (serial) search (seconds / item).

Conjunction / absence searches can require 20-50 ms per item.

Citation: Treisman, A. M. & Gelade, G. (1980). A feature-integration
theory of attention. *Cognitive Psychology*, 12(1), 97-136.
"""

# ═══════════════════════════════════════════════════════════════════════════
# Motor timing
# ═══════════════════════════════════════════════════════════════════════════

MOTOR_PREPARATION_TIME: Interval = Interval(0.100, 0.200)
"""Motor preparation time (seconds).

Time from stimulus identification to movement onset.

Citation: Card, S. K., Moran, T. P., & Newell, A. (1983). *The
Psychology of Human-Computer Interaction*. Lawrence Erlbaum.
"""

MOTOR_EXECUTION_TIME: Interval = Interval(0.070, 0.150)
"""Motor execution time per keystroke or click (seconds).

Ranges from expert typist (70 ms) to hunt-and-peck novice (150 ms).

Citation: Card, S. K. et al. (1983). Keystroke-Level Model.
"""

# ═══════════════════════════════════════════════════════════════════════════
# Perceptual timing
# ═══════════════════════════════════════════════════════════════════════════

PERCEPTION_TIME_VISUAL: Interval = Interval(0.080, 0.150)
"""Visual perceptual encoding time (seconds).

Approximate time for foveal identification of a familiar icon or word.

Citation: Card, S. K. et al. (1983). Model Human Processor.
"""

PERCEPTION_TIME_AUDITORY: Interval = Interval(0.050, 0.100)
"""Auditory perceptual encoding time (seconds).

Auditory stimuli are processed faster than visual ones.

Citation: Card, S. K. et al. (1983). Model Human Processor.
"""

# ═══════════════════════════════════════════════════════════════════════════
# Eye movements
# ═══════════════════════════════════════════════════════════════════════════

SACCADE_DURATION: Interval = Interval(0.020, 0.050)
"""Saccade duration (seconds).

Duration of a single saccadic eye movement (typically 20-50 ms
depending on amplitude).

Citation: Rayner, K. (1998). Eye movements in reading and information
processing: 20 years of research. *Psychological Bulletin*, 124(3),
372-422.
"""

FIXATION_DURATION: Interval = Interval(0.150, 0.400)
"""Fixation duration (seconds).

Mean fixation during reading is ~200-250 ms; visual search fixations
can be up to 400 ms.

Citation: Rayner, K. (1998).
"""

# ═══════════════════════════════════════════════════════════════════════════
# Display / conversion
# ═══════════════════════════════════════════════════════════════════════════

DISPLAY_PPI: float = 96.0
"""Default display resolution (pixels per inch).

Standard CSS reference pixel density.
"""

PIXEL_TO_MM: float = 25.4 / DISPLAY_PPI
"""Conversion factor: multiply pixel distance by this to get millimetres.

Derived from ``25.4 mm/inch / PPI``.  At 96 PPI this is approx 0.2646 mm/px.
"""

# ═══════════════════════════════════════════════════════════════════════════
# Channel capacities (bits/s)
# ═══════════════════════════════════════════════════════════════════════════

CHANNEL_CAPACITIES: Dict[str, Interval] = {
    "visual": Interval(30.0, 50.0),
    "auditory": Interval(20.0, 35.0),
    "haptic": Interval(2.0, 8.0),
    "motor_hand": Interval(8.0, 12.0),
    "motor_eye": Interval(2.0, 5.0),
    "motor_voice": Interval(30.0, 50.0),
}
"""Information channel capacities (bits / second).

Approximate throughput limits for each perceptual / motor channel,
compiled from multiple-resource theory.

Citation: Wickens, C. D. (2002). Multiple resources and performance
prediction. *Theoretical Issues in Ergonomics Science*, 3(2), 159-177.
"""

# ═══════════════════════════════════════════════════════════════════════════
# UI sizing guidelines
# ═══════════════════════════════════════════════════════════════════════════

MINIMUM_TARGET_SIZE_PX: float = 24.0
"""Minimum interactive target size (pixels).

Below this size, error rates increase sharply.

Citation: WCAG 2.2, Success Criterion 2.5.8 -- Target Size (Minimum).
"""

RECOMMENDED_TARGET_SIZE_PX: float = 44.0
"""Recommended interactive target size (pixels).

Apple HIG and Material Design both recommend >= 44 px (approx 7 mm at 160 ppi).

Citation: Apple Human Interface Guidelines; Google Material Design;
ISO 9241-420:2011.
"""

# ═══════════════════════════════════════════════════════════════════════════
# Hick's Law practical limits
# ═══════════════════════════════════════════════════════════════════════════

MAX_REASONABLE_CHOICES: int = 12
"""Practical upper limit on simultaneous choices for Hick's Law.

Beyond ~12 items, users typically switch to visual search or hierarchical
navigation rather than scanning all options.

Citation: Landauer, T. K. & Nachbar, D. W. (1985). Selection from
alphabetic and numeric menu trees using a touch screen. *Proceedings of
CHI '85*, 73-78.
"""

# ═══════════════════════════════════════════════════════════════════════════
# Rationality parameter beta
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_BETA_RANGE: Interval = Interval(0.1, 20.0)
"""Default range for the rationality parameter beta.

beta -> 0 represents a fully random agent; beta -> inf a perfectly rational one.
Typical human behaviour is captured in the range [1, 10].

Citation: Ortega, P. A. & Braun, D. A. (2013). Thermodynamics as a
theory of decision-making with information-processing costs. *Proceedings
of the Royal Society A*, 469(2153), 20120683.
"""

# ═══════════════════════════════════════════════════════════════════════════
# Population percentiles for motor/cognitive parameters
# ═══════════════════════════════════════════════════════════════════════════

POPULATION_PERCENTILES: Dict[str, Dict[str, float]] = {
    "fitts_b": {
        "p5": 0.105,
        "p25": 0.130,
        "p50": 0.155,
        "p75": 0.190,
        "p95": 0.240,
    },
    "hick_b": {
        "p5": 0.105,
        "p25": 0.125,
        "p50": 0.150,
        "p75": 0.175,
        "p95": 0.195,
    },
    "working_memory_capacity": {
        "p5": 2.5,
        "p25": 3.5,
        "p50": 4.0,
        "p75": 5.0,
        "p95": 6.5,
    },
    "visual_search_slope": {
        "p5": 0.012,
        "p25": 0.020,
        "p50": 0.028,
        "p75": 0.038,
        "p95": 0.048,
    },
    "motor_execution_time": {
        "p5": 0.072,
        "p25": 0.090,
        "p50": 0.110,
        "p75": 0.130,
        "p95": 0.148,
    },
    "perception_time_visual": {
        "p5": 0.082,
        "p25": 0.095,
        "p50": 0.110,
        "p75": 0.130,
        "p95": 0.148,
    },
}
"""Population percentile tables (5th, 25th, 50th, 75th, 95th) for key
cognitive and motor parameters.

Values are approximate, compiled from multiple sources.  They allow
the oracle to evaluate usability for a *range* of users rather than
a single median user.

Citations:
- Card, S. K. et al. (1983). Model Human Processor.
- MacKenzie, I. S. (1992). Fitts' law as a research and design tool.
- Hyman, R. (1953). Stimulus information as a determinant of reaction time.
- Cowan, N. (2001). The magical number 4 in short-term memory.
"""

# ═══════════════════════════════════════════════════════════════════════════
# Derived convenience constants
# ═══════════════════════════════════════════════════════════════════════════

FITTS_IP_RANGE: Interval = Interval(
    1.0 / FITTS_B_RANGE.high, 1.0 / FITTS_B_RANGE.low
)
"""Fitts' Index of Performance (bits/s) = 1/b.

Ranges from ~4 bits/s (slow) to ~10 bits/s (fast).
"""

MINIMUM_TARGET_SIZE_MM: float = MINIMUM_TARGET_SIZE_PX * PIXEL_TO_MM
"""Minimum target size in millimetres (derived from PPI)."""

RECOMMENDED_TARGET_SIZE_MM: float = RECOMMENDED_TARGET_SIZE_PX * PIXEL_TO_MM
"""Recommended target size in millimetres (derived from PPI)."""

WORKING_MEMORY_DECAY_RATE_RANGE: Interval = Interval(
    0.693 / WORKING_MEMORY_DECAY_HALF_LIFE.high,
    0.693 / WORKING_MEMORY_DECAY_HALF_LIFE.low,
)
"""Working-memory decay rate lambda (1/s) = ln(2) / half_life.

Derived from the half-life range.
"""

HICK_BITS_FOR_MAX_CHOICES: float = 3.70  # log2(12 + 1) ~ 3.70
"""Number of bits of entropy for MAX_REASONABLE_CHOICES (= 12).

log2(12 + 1) ~ 3.70 bits.  Beyond this, Hick's law loses validity.
"""


__all__ = [
    "FITTS_A_RANGE",
    "FITTS_B_RANGE",
    "FITTS_IP_RANGE",
    "HICK_A_RANGE",
    "HICK_B_RANGE",
    "HICK_BITS_FOR_MAX_CHOICES",
    "WORKING_MEMORY_CAPACITY",
    "WORKING_MEMORY_DECAY_HALF_LIFE",
    "WORKING_MEMORY_DECAY_RATE_RANGE",
    "VISUAL_SEARCH_EFFICIENT_SLOPE",
    "VISUAL_SEARCH_INEFFICIENT_SLOPE",
    "MOTOR_PREPARATION_TIME",
    "MOTOR_EXECUTION_TIME",
    "PERCEPTION_TIME_VISUAL",
    "PERCEPTION_TIME_AUDITORY",
    "SACCADE_DURATION",
    "FIXATION_DURATION",
    "PIXEL_TO_MM",
    "DISPLAY_PPI",
    "CHANNEL_CAPACITIES",
    "MINIMUM_TARGET_SIZE_PX",
    "MINIMUM_TARGET_SIZE_MM",
    "RECOMMENDED_TARGET_SIZE_PX",
    "RECOMMENDED_TARGET_SIZE_MM",
    "MAX_REASONABLE_CHOICES",
    "DEFAULT_BETA_RANGE",
    "POPULATION_PERCENTILES",
]
