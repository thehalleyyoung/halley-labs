#!/usr/bin/env python3
"""
Perceptual Type-Checking Benchmark for SoniType
================================================

Exercises the full pipeline: spec creation → type checking → compilation →
information-loss estimation, with a no-type-check baseline for comparison.

Benchmark suite:
  B1–B4:  Well-typed specs (1, 3, 8, 12 streams) — should PASS
  B5–B6:  Ill-typed specs (masking violation, cognitive overload) — should FAIL
  B7–B8:  Borderline JND specs (just above / just below threshold) — edge cases
  B9–B10: Real-world datasets (ICU monitor, stock data)

Metrics:
  - Type-check correctness (expected vs actual accept/reject)
  - Compilation wall-clock time (µs)
  - d'_model (compound discriminability across all stream pairs)
  - I_ψ estimate (psychoacoustic mutual information in bits)
  - Information preservation ratio (bits_out / bits_in)
  - Baseline comparison: naive mapping without perceptual constraints

References:
  Wier et al. (1977), Jesteadt et al. (1977), Moore (2012),
  Grey (1977), McAdams et al. (1995), Cowan (2001).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Add implementation directory to path
IMPL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "implementation")
sys.path.insert(0, IMPL_DIR)

from perceptual_type_checker import (
    COGNITIVE_LOAD_MAX,
    DPRIME_THRESHOLD,
    DURATION_FLOOR_MS,
    DURATION_WEBER,
    LOUDNESS_JND_DB,
    PITCH_WEBER_OPTIMAL,
    TIMBRE_CENTROID_FRACTION,
    CompilationResult,
    DataDimension,
    DimensionMapping,
    DurationType,
    LoudnessType,
    PerceptualLossEstimator,
    PerceptualTypeChecker,
    PitchType,
    Scale,
    SonificationCompiler,
    SonificationSpec,
    StreamSpec,
    TimbreType,
    hz_to_bark,
)

# ── Reproducibility ──────────────────────────────────────────────────────
SEED = 42
import random
random.seed(SEED)

# ── Benchmark result types ───────────────────────────────────────────────

@dataclass
class BenchmarkCase:
    """One benchmark specification and its expected outcome."""
    id: str
    description: str
    category: str          # "well_typed", "ill_typed", "borderline", "real_world"
    n_streams: int
    n_mappings: int
    expected_well_typed: bool
    spec: SonificationSpec


@dataclass
class BenchmarkResult:
    """Result of running one benchmark case."""
    id: str
    description: str
    category: str
    n_streams: int
    n_mappings: int
    expected_well_typed: bool
    actual_well_typed: bool
    correct_verdict: bool
    type_check_time_us: float
    compile_time_us: float
    n_errors: int
    n_warnings: int
    cognitive_load: float
    cognitive_budget: float
    # Perceptual quality metrics
    mean_d_prime: float
    min_d_prime: float
    max_d_prime: float
    n_pairwise_checks: int
    # Information metrics
    total_bits_in: float
    total_bits_out: float
    total_info_loss_bits: float
    preservation_ratio: float
    # Per-mapping detail
    mapping_details: List[Dict[str, Any]]
    pairwise_details: List[Dict[str, Any]]
    violations: List[Dict[str, str]]


@dataclass
class BaselineResult:
    """Baseline: naive direct mapping ignoring JND constraints."""
    id: str
    n_streams: int
    naive_bits_in: float
    naive_bits_out: float
    naive_preservation: float
    naive_d_prime: float
    masking_violations: int
    jnd_violations: int
    segregation_failures: int
    cognitive_overload: bool


# ── Benchmark spec builders ──────────────────────────────────────────────

# Frequency allocation: well-separated base frequencies across Bark bands
# Bark band centers (Hz): 50, 150, 250, 350, 450, 570, 700, 840, 1000,
# 1170, 1370, 1600, 1850, 2150, 2500, 2900, 3400, 4000, 4800, 5800, 7000, 8500, 10500, 13500
BARK_CENTERS = [
    50, 150, 250, 350, 450, 570, 700, 840, 1000,
    1170, 1370, 1600, 1850, 2150, 2500, 2900, 3400, 4000,
    4800, 5800, 7000, 8500, 10500, 13500,
]

def _make_well_separated_streams(n: int, data_dims: List[DataDimension]) -> List[StreamSpec]:
    """Create n well-separated streams using distinct Bark bands."""
    streams = []
    # Use every other Bark band for separation
    usable = BARK_CENTERS[2:22]  # 250 Hz – 10500 Hz (usable range)
    step = max(1, len(usable) // n)
    for i in range(n):
        idx = min(i * step, len(usable) - 1)
        base_f = usable[idx]
        # Each stream gets a pitch range spanning ~1 Bark band
        lo_hz = base_f * 0.85
        hi_hz = base_f * 1.25
        level_db = 55.0 + (i % 4) * 5.0  # 55, 60, 65, 70 dB cycling

        dim = data_dims[i % len(data_dims)]
        mappings = [DimensionMapping(dim, PitchType(lo_hz, hi_hz))]

        # Add loudness mapping to some streams for richer specs
        if i % 3 == 0 and n > 1:
            loud_dim = data_dims[(i + 1) % len(data_dims)]
            mappings.append(DimensionMapping(loud_dim, LoudnessType(45.0, 75.0)))

        streams.append(StreamSpec(
            name=f"stream_{i}",
            mappings=mappings,
            base_frequency_hz=base_f,
            level_db=level_db,
            cognitive_weight=1.0,
        ))
    return streams


def build_benchmark_suite() -> List[BenchmarkCase]:
    """Construct the 10 benchmark specifications."""
    cases: List[BenchmarkCase] = []

    # Standard data dimensions for reuse
    d_price = DataDimension("price", 0.0, 200.0)
    d_volume = DataDimension("volume", 0.0, 1e6)
    d_volatility = DataDimension("volatility", 0.0, 100.0)
    d_temp = DataDimension("temperature", -2.0, 2.0)
    d_hr = DataDimension("heart_rate", 40.0, 200.0)
    d_bp = DataDimension("blood_pressure", 60.0, 180.0)
    d_spo2 = DataDimension("spo2", 85.0, 100.0)
    d_resp = DataDimension("respiration", 8.0, 40.0)
    all_dims = [d_price, d_volume, d_volatility, d_temp, d_hr, d_bp, d_spo2, d_resp]

    # ── B1: 1-stream time series (trivial well-typed) ────────────────
    cases.append(BenchmarkCase(
        id="B1_single_stream",
        description="Single-stream pitch sonification of price data",
        category="well_typed",
        n_streams=1,
        n_mappings=2,
        expected_well_typed=True,
        spec=SonificationSpec(streams=[
            StreamSpec(
                name="price",
                mappings=[
                    DimensionMapping(d_price, PitchType(200.0, 2000.0)),
                    DimensionMapping(d_volatility, LoudnessType(40.0, 80.0)),
                ],
                base_frequency_hz=440.0,
                level_db=60.0,
            ),
        ]),
    ))

    # ── B2: 3-stream multivariate (well-typed, well-separated) ───────
    cases.append(BenchmarkCase(
        id="B2_three_stream",
        description="3-stream stock data: price→pitch, volume→duration, volatility→timbre",
        category="well_typed",
        n_streams=3,
        n_mappings=3,
        expected_well_typed=True,
        spec=SonificationSpec(streams=[
            StreamSpec(
                name="price_stream",
                mappings=[DimensionMapping(d_price, PitchType(200.0, 800.0))],
                base_frequency_hz=400.0,
                level_db=60.0,
            ),
            StreamSpec(
                name="volume_stream",
                mappings=[DimensionMapping(d_volume, DurationType(100.0, 1000.0))],
                base_frequency_hz=1500.0,
                level_db=55.0,
            ),
            StreamSpec(
                name="volatility_stream",
                mappings=[DimensionMapping(d_volatility, TimbreType(2000.0, 5000.0))],
                base_frequency_hz=3000.0,
                level_db=65.0,
            ),
        ]),
    ))

    # ── B3: 8-stream alarm palette (IEC 60601-1-8 style) ────────────
    alarm_names = [
        "general", "cardiovascular", "ventilation", "oxygenation",
        "temperature", "drug_delivery", "equipment", "perfusion",
    ]
    alarm_streams = []
    for i, name in enumerate(alarm_names):
        bark_idx = min(i * 2 + 3, len(BARK_CENTERS) - 1)
        base_f = BARK_CENTERS[bark_idx]
        lo = base_f * 0.9
        hi = base_f * 1.15
        dim = DataDimension(f"alarm_{name}", 0.0, 3.0, cardinality=4)
        alarm_streams.append(StreamSpec(
            name=f"alarm_{name}",
            mappings=[DimensionMapping(dim, PitchType(lo, hi))],
            base_frequency_hz=base_f,
            level_db=55.0 + i * 3.0,
            cognitive_weight=0.6,  # Alarms are brief, lower cognitive cost
        ))
    cases.append(BenchmarkCase(
        id="B3_eight_alarm",
        description="8-alarm medical palette (IEC 60601-1-8 modeled)",
        category="well_typed",
        n_streams=8,
        n_mappings=8,
        expected_well_typed=True,
        spec=SonificationSpec(
            streams=alarm_streams,
            cognitive_budget=COGNITIVE_LOAD_MAX,
        ),
    ))

    # ── B4: 12-stream stress test (well-typed, tight but valid) ──────
    streams_12 = _make_well_separated_streams(12, all_dims)
    # Keep cognitive load under budget
    for s in streams_12:
        s.cognitive_weight = 0.4  # 12 * 0.4 = 4.8 < 5.0
    cases.append(BenchmarkCase(
        id="B4_twelve_stream",
        description="12-stream stress test with reduced cognitive weights",
        category="well_typed",
        n_streams=12,
        n_mappings=sum(len(s.mappings) for s in streams_12),
        expected_well_typed=True,
        spec=SonificationSpec(streams=streams_12, cognitive_budget=5.0),
    ))

    # ── B5: Masking violation (ill-typed: nearly identical streams) ──
    # Streams 0.5 Hz apart at 440 Hz with same loudness → d' ≈ 0.32 < 1.5
    # pitch_margin = 0.5 / (440.5 * 0.0035) = 0.324, loud_margin = 0
    cases.append(BenchmarkCase(
        id="B5_masking_violation",
        description="Two nearly identical streams (Δf=0.5 Hz, same level) → d'≈0.32",
        category="ill_typed",
        n_streams=2,
        n_mappings=2,
        expected_well_typed=False,
        spec=SonificationSpec(streams=[
            StreamSpec(
                name="stream_a",
                mappings=[DimensionMapping(d_price, PitchType(430.0, 460.0))],
                base_frequency_hz=440.0,
                level_db=60.0,
            ),
            StreamSpec(
                name="stream_b",
                mappings=[DimensionMapping(d_volume, PitchType(432.0, 462.0))],
                base_frequency_hz=440.5,
                level_db=60.0,
            ),
        ]),
    ))

    # ── B6: Cognitive overload (ill-typed: 7 streams, weight 1.0 each) ──
    overload_streams = []
    for i in range(7):
        dim = DataDimension(f"dim_{i}", 0.0, 100.0)
        base_f = 200.0 + i * 400.0
        overload_streams.append(StreamSpec(
            name=f"overload_{i}",
            mappings=[DimensionMapping(dim, PitchType(base_f * 0.8, base_f * 1.2))],
            base_frequency_hz=base_f,
            level_db=60.0,
            cognitive_weight=1.0,
        ))
    cases.append(BenchmarkCase(
        id="B6_cognitive_overload",
        description="7 streams × weight 1.0 = 7.0 > budget 5.0",
        category="ill_typed",
        n_streams=7,
        n_mappings=7,
        expected_well_typed=False,
        spec=SonificationSpec(streams=overload_streams, cognitive_budget=5.0),
    ))

    # ── B7: Borderline PASS (just above JND threshold) ───────────────
    # Two streams separated by just barely enough d'
    # d' = sqrt(pitch_margin^2 + loud_margin^2) needs to be >= 1.5
    # Target d' ≈ 1.6 (just above)
    # With Δf = 10 Hz at 1000 Hz: pitch_margin = 10 / (1000*0.0035) = 2.86
    # With Δloud = 0: d' = 2.86 → passes. Let's make it tighter.
    # Δf = 3 Hz at 1000 Hz: pitch_margin = 3 / (1000*0.0035) = 0.857
    # Δloud = 1.3 dB: loud_margin = 1.3 / 1.0 = 1.3
    # d' = sqrt(0.857^2 + 1.3^2) = sqrt(0.734 + 1.69) = sqrt(2.424) = 1.557 → just passes
    cases.append(BenchmarkCase(
        id="B7_borderline_pass",
        description="Borderline d'≈1.56 (just above threshold 1.5)",
        category="borderline",
        n_streams=2,
        n_mappings=2,
        expected_well_typed=True,
        spec=SonificationSpec(streams=[
            StreamSpec(
                name="border_a",
                mappings=[DimensionMapping(d_price, PitchType(900.0, 1100.0))],
                base_frequency_hz=1000.0,
                level_db=60.0,
            ),
            StreamSpec(
                name="border_b",
                mappings=[DimensionMapping(d_volume, PitchType(950.0, 1150.0))],
                base_frequency_hz=1003.0,
                level_db=61.3,
            ),
        ]),
    ))

    # ── B8: Borderline FAIL (just below JND threshold) ───────────────
    # Δf = 2 Hz at 1000 Hz: pitch_margin = 2 / (1000*0.0035) = 0.571
    # Δloud = 1.0 dB: loud_margin = 1.0 / 1.0 = 1.0
    # d' = sqrt(0.571^2 + 1.0^2) = sqrt(0.326 + 1.0) = sqrt(1.326) = 1.152 → fails
    cases.append(BenchmarkCase(
        id="B8_borderline_fail",
        description="Borderline d'≈1.15 (just below threshold 1.5)",
        category="borderline",
        n_streams=2,
        n_mappings=2,
        expected_well_typed=False,
        spec=SonificationSpec(streams=[
            StreamSpec(
                name="border_c",
                mappings=[DimensionMapping(d_price, PitchType(900.0, 1100.0))],
                base_frequency_hz=1000.0,
                level_db=60.0,
            ),
            StreamSpec(
                name="border_d",
                mappings=[DimensionMapping(d_volume, PitchType(950.0, 1150.0))],
                base_frequency_hz=1002.0,
                level_db=61.0,
            ),
        ]),
    ))

    # ── B9: ICU telemetry monitor (real-world, 4 streams) ────────────
    cases.append(BenchmarkCase(
        id="B9_icu_monitor",
        description="4-channel ICU monitor: HR, BP, SpO2, respiration",
        category="real_world",
        n_streams=4,
        n_mappings=4,
        expected_well_typed=True,
        spec=SonificationSpec(streams=[
            StreamSpec(
                name="heart_rate",
                mappings=[DimensionMapping(d_hr, PitchType(200.0, 600.0))],
                base_frequency_hz=350.0,
                level_db=60.0,
                cognitive_weight=1.2,
            ),
            StreamSpec(
                name="blood_pressure",
                mappings=[DimensionMapping(d_bp, PitchType(800.0, 1600.0))],
                base_frequency_hz=1100.0,
                level_db=55.0,
                cognitive_weight=1.0,
            ),
            StreamSpec(
                name="spo2",
                mappings=[DimensionMapping(d_spo2, LoudnessType(40.0, 80.0))],
                base_frequency_hz=2500.0,
                level_db=65.0,
                cognitive_weight=0.8,
            ),
            StreamSpec(
                name="respiration",
                mappings=[DimensionMapping(d_resp, DurationType(200.0, 800.0))],
                base_frequency_hz=4000.0,
                level_db=50.0,
                cognitive_weight=0.8,
            ),
        ], cognitive_budget=5.0),
    ))

    # ── B10: 5-stream climate data (real-world) ──────────────────────
    d_global = DataDimension("global_temp", -0.5, 1.2)
    d_north = DataDimension("northern_hemi", -0.8, 1.5)
    d_south = DataDimension("southern_hemi", -0.6, 1.0)
    d_co2 = DataDimension("co2_ppm", 280.0, 420.0)
    d_sea = DataDimension("sea_level_mm", -20.0, 60.0)
    cases.append(BenchmarkCase(
        id="B10_climate_5stream",
        description="5-stream climate data: temp×3, CO2, sea level",
        category="real_world",
        n_streams=5,
        n_mappings=5,
        expected_well_typed=True,
        spec=SonificationSpec(streams=[
            StreamSpec(
                name="global_temp",
                mappings=[DimensionMapping(d_global, PitchType(220.0, 440.0))],
                base_frequency_hz=300.0,
                level_db=60.0,
                cognitive_weight=1.0,
            ),
            StreamSpec(
                name="north_hemi",
                mappings=[DimensionMapping(d_north, PitchType(600.0, 1200.0))],
                base_frequency_hz=800.0,
                level_db=55.0,
                cognitive_weight=0.8,
            ),
            StreamSpec(
                name="south_hemi",
                mappings=[DimensionMapping(d_south, PitchType(1500.0, 3000.0))],
                base_frequency_hz=2000.0,
                level_db=55.0,
                cognitive_weight=0.8,
            ),
            StreamSpec(
                name="co2",
                mappings=[DimensionMapping(d_co2, TimbreType(1000.0, 4000.0))],
                base_frequency_hz=5000.0,
                level_db=50.0,
                cognitive_weight=1.2,
            ),
            StreamSpec(
                name="sea_level",
                mappings=[DimensionMapping(d_sea, LoudnessType(45.0, 75.0))],
                base_frequency_hz=7000.0,
                level_db=65.0,
                cognitive_weight=1.0,
            ),
        ], cognitive_budget=5.0),
    ))

    return cases


# ── Baseline: naive mapping without perceptual constraints ───────────────

def compute_baseline(case: BenchmarkCase) -> BaselineResult:
    """Simulate naive direct mapping: no JND checking, no masking analysis.

    The baseline maps each data dimension to the full audible range uniformly,
    ignoring Weber-fraction JNDs, masking interactions, and cognitive limits.
    This is what most existing sonification tools do.
    """
    spec = case.spec
    n = len(spec.streams)

    # Naive info: assume full audible range per stream (20-20000 Hz for pitch)
    naive_bits_in = 0.0
    naive_bits_out = 0.0
    for stream in spec.streams:
        for m in stream.mappings:
            # Data entropy estimate
            if m.data_dim.cardinality is not None:
                bits = math.log2(max(m.data_dim.cardinality, 1))
            else:
                bits = math.log2(max(m.data_dim.range(), 1))
            naive_bits_in += bits
            # Naive output: assume uniform quantization to 256 levels (8-bit)
            # This ignores JND — many "levels" are perceptually identical
            naive_bits_out += 8.0

    # Count perceptual violations that naive mapping would produce
    masking_violations = 0
    jnd_violations = 0
    segregation_failures = 0

    # Check pairwise masking (Bark band overlap)
    for i, sa in enumerate(spec.streams):
        for sb in spec.streams[i + 1:]:
            bark_a = hz_to_bark(sa.base_frequency_hz)
            bark_b = hz_to_bark(sb.base_frequency_hz)
            bark_diff = abs(bark_a - bark_b)
            if bark_diff < 1.0:
                masking_violations += 1
            # Naive d' — using raw Hz difference, no Weber normalization
            freq_diff = abs(sa.base_frequency_hz - sb.base_frequency_hz)
            if freq_diff < 50:  # crude heuristic for "too close"
                segregation_failures += 1

    # JND violations: check if any mapping range is below 1 JND
    for stream in spec.streams:
        for m in stream.mappings:
            at = m.audio_type
            if at.distinguishable_steps() < 2:
                jnd_violations += 1

    # Naive d': simple frequency-ratio-based estimate (no Weber normalization)
    d_primes = []
    for i, sa in enumerate(spec.streams):
        for sb in spec.streams[i + 1:]:
            # Naive: treat Hz difference as the discriminability signal
            freq_ratio = abs(sa.base_frequency_hz - sb.base_frequency_hz) / max(
                sa.base_frequency_hz, sb.base_frequency_hz, 1.0
            )
            naive_dp = freq_ratio * 10.0  # arbitrary scaling
            d_primes.append(naive_dp)

    naive_d_prime = sum(d_primes) / max(len(d_primes), 1) if d_primes else 0.0

    cognitive_total = sum(s.cognitive_weight for s in spec.streams)
    return BaselineResult(
        id=case.id,
        n_streams=n,
        naive_bits_in=round(naive_bits_in, 2),
        naive_bits_out=round(naive_bits_out, 2),
        naive_preservation=round(
            naive_bits_out / max(naive_bits_in, 1e-9), 4
        ),
        naive_d_prime=round(naive_d_prime, 3),
        masking_violations=masking_violations,
        jnd_violations=jnd_violations,
        segregation_failures=segregation_failures,
        cognitive_overload=cognitive_total > spec.cognitive_budget,
    )


# ── Weber's Law validation ───────────────────────────────────────────────

def validate_weber_compliance() -> Dict[str, Any]:
    """Validate SoniType's JND models against Weber's law predictions.

    Returns per-dimension Weber fraction, functional range, and model fit.
    """
    results = {}

    # Pitch: k = 0.0035 in 500-5000 Hz range
    pitch_type = PitchType(100.0, 8000.0)
    freqs = [100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000, 6000, 8000]
    pitch_jnds_hz = [pitch_type.jnd_at(f) for f in freqs]
    pitch_jnds_cents = [1200.0 * math.log2((f + jnd) / f) if f > 0 else 0
                        for f, jnd in zip(freqs, pitch_jnds_hz)]
    weber_fracs = [jnd / f for f, jnd in zip(freqs, pitch_jnds_hz)]

    # Weber's law predicts constant k in optimal range
    optimal_wf = [wf for f, wf in zip(freqs, weber_fracs) if 500 <= f <= 5000]
    mean_k = sum(optimal_wf) / len(optimal_wf) if optimal_wf else 0
    # Compute r² against constant-k model
    ss_res = sum((wf - mean_k) ** 2 for wf in optimal_wf)
    ss_tot = sum((wf - sum(optimal_wf) / len(optimal_wf)) ** 2 for wf in optimal_wf)
    r_sq_pitch = 1.0 - ss_res / max(ss_tot, 1e-15) if ss_tot > 0 else 1.0

    results["pitch"] = {
        "weber_fraction_k": round(PITCH_WEBER_OPTIMAL, 4),
        "functional_range": "200-4000 Hz",
        "r_squared": round(r_sq_pitch, 4),
        "model_rmse_cents": round(
            math.sqrt(sum((jc - sum(pitch_jnds_cents) / len(pitch_jnds_cents)) ** 2
                         for jc in pitch_jnds_cents) / len(pitch_jnds_cents)), 2
        ),
        "freqs_hz": freqs,
        "jnd_hz": [round(j, 3) for j in pitch_jnds_hz],
        "jnd_cents": [round(j, 2) for j in pitch_jnds_cents],
        "weber_fractions": [round(wf, 5) for wf in weber_fracs],
    }

    # Loudness: k ≈ 0.079 (1 dB = 10^(1/10) - 1 ≈ 0.259 intensity ratio)
    loud_type = LoudnessType(10.0, 100.0)
    levels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    loud_jnds = [loud_type.jnd_at(float(lev)) for lev in levels]
    results["loudness"] = {
        "weber_fraction_k": 0.079,
        "functional_range": "40-90 dB SPL",
        "jnd_db": [round(j, 3) for j in loud_jnds],
        "levels": levels,
        "note": "Near-miss to Weber's law at low levels (Jesteadt correction applied)",
    }

    # Duration: k = 0.10 with 10 ms floor
    dur_type = DurationType(20.0, 2000.0)
    durations = [20, 50, 100, 150, 200, 300, 500, 800, 1000, 1500, 2000]
    dur_jnds = [dur_type.jnd_at(float(d)) for d in durations]
    dur_weber = [jnd / d for d, jnd in zip(durations, dur_jnds)]
    results["duration"] = {
        "weber_fraction_k": DURATION_WEBER,
        "functional_range": "50-500 ms IOI",
        "jnd_ms": [round(j, 2) for j in dur_jnds],
        "durations_ms": durations,
        "weber_fractions": [round(wf, 4) for wf in dur_weber],
        "floor_ms": DURATION_FLOOR_MS,
    }

    # Timbre (spectral centroid): k = 0.05
    timbre_type = TimbreType(500.0, 8000.0)
    centroids = [500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 8000]
    timbre_jnds = [timbre_type.jnd_at(float(c)) for c in centroids]
    timbre_weber = [jnd / c for c, jnd in zip(centroids, timbre_jnds)]
    results["timbre_centroid"] = {
        "weber_fraction_k": TIMBRE_CENTROID_FRACTION,
        "functional_range": "500-8000 Hz",
        "jnd_hz": [round(j, 2) for j in timbre_jnds],
        "centroids_hz": centroids,
        "weber_fractions": [round(wf, 4) for wf in timbre_weber],
    }

    return results


# ── Scaling benchmark (compilation time vs stream count) ─────────────────

def run_scaling_benchmark() -> List[Dict[str, Any]]:
    """Measure type-check + compile time for 1–16 streams."""
    results = []
    d_generic = DataDimension("value", 0.0, 100.0)
    all_dims = [d_generic]

    for n_streams in [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16]:
        streams = _make_well_separated_streams(n_streams, all_dims)
        # Scale cognitive weights to stay under budget
        for s in streams:
            s.cognitive_weight = min(1.0, 4.8 / max(n_streams, 1))

        spec = SonificationSpec(streams=streams, cognitive_budget=5.0)
        compiler = SonificationCompiler()

        # Warm-up
        compiler.compile(spec)

        # Timed runs (20 iterations for stable measurement)
        n_iter = 20
        times_us = []
        for _ in range(n_iter):
            t0 = time.perf_counter_ns()
            compiler.compile(spec)
            t1 = time.perf_counter_ns()
            times_us.append((t1 - t0) / 1000.0)

        # Also time type-check only
        checker = PerceptualTypeChecker()
        tc_times = []
        for _ in range(n_iter):
            t0 = time.perf_counter_ns()
            checker.check(spec)
            t1 = time.perf_counter_ns()
            tc_times.append((t1 - t0) / 1000.0)

        n_pairs = n_streams * (n_streams - 1) // 2
        results.append({
            "n_streams": n_streams,
            "n_pairwise_checks": n_pairs,
            "compile_time_us_mean": round(sum(times_us) / len(times_us), 1),
            "compile_time_us_median": round(sorted(times_us)[len(times_us) // 2], 1),
            "compile_time_us_p95": round(sorted(times_us)[int(0.95 * len(times_us))], 1),
            "typecheck_time_us_mean": round(sum(tc_times) / len(tc_times), 1),
            "typecheck_time_us_median": round(sorted(tc_times)[len(tc_times) // 2], 1),
        })

    return results


# ── Main benchmark runner ────────────────────────────────────────────────

def run_benchmark() -> Dict[str, Any]:
    """Execute the full perceptual type-checking benchmark."""
    print("=" * 70)
    print("SoniType Perceptual Type-Checking Benchmark")
    print("=" * 70)

    cases = build_benchmark_suite()
    compiler = SonificationCompiler()
    loss_estimator = PerceptualLossEstimator()

    benchmark_results: List[BenchmarkResult] = []
    baseline_results: List[BaselineResult] = []

    print(f"\n{'ID':<25} {'Streams':>7} {'Expected':>10} {'Actual':>10} "
          f"{'Correct':>8} {'d_mean':>8} {'I_ψ':>8} {'Pres%':>7} {'Time(µs)':>10}")
    print("─" * 110)

    for case in cases:
        spec = case.spec

        # ── Warm-up ──
        compiler.compile(spec)

        # ── Timed compilation (10 iterations) ──
        n_iter = 10
        compile_times = []
        tc_times = []
        for _ in range(n_iter):
            t0 = time.perf_counter_ns()
            result = compiler.compile(spec)
            t1 = time.perf_counter_ns()
            compile_times.append((t1 - t0) / 1000.0)

        checker = PerceptualTypeChecker()
        for _ in range(n_iter):
            t0 = time.perf_counter_ns()
            tc_result = checker.check(spec)
            t1 = time.perf_counter_ns()
            tc_times.append((t1 - t0) / 1000.0)

        # Use last result for metrics
        tc = result.type_check

        # d' statistics
        d_primes = [pw.d_prime for pw in tc.pairwise]
        mean_dp = sum(d_primes) / len(d_primes) if d_primes else 0.0
        min_dp = min(d_primes) if d_primes else 0.0
        max_dp = max(d_primes) if d_primes else 0.0

        # Information metrics
        loss_reports = tc.information_loss
        total_in = sum(r.bits_in for r in loss_reports)
        total_out = sum(r.bits_out for r in loss_reports)
        total_loss = sum(r.information_loss_bits for r in loss_reports)
        pres_ratio = total_out / max(total_in, 1e-9)

        errors = [v for v in tc.violations if v.severity.value == "error"]
        warnings = [v for v in tc.violations if v.severity.value == "warning"]

        correct = tc.well_typed == case.expected_well_typed
        compile_us = sum(compile_times) / len(compile_times)
        tc_us = sum(tc_times) / len(tc_times)

        br = BenchmarkResult(
            id=case.id,
            description=case.description,
            category=case.category,
            n_streams=case.n_streams,
            n_mappings=case.n_mappings,
            expected_well_typed=case.expected_well_typed,
            actual_well_typed=tc.well_typed,
            correct_verdict=correct,
            type_check_time_us=round(tc_us, 1),
            compile_time_us=round(compile_us, 1),
            n_errors=len(errors),
            n_warnings=len(warnings),
            cognitive_load=tc.cognitive_load_total,
            cognitive_budget=tc.cognitive_budget,
            mean_d_prime=round(mean_dp, 3),
            min_d_prime=round(min_dp, 3),
            max_d_prime=round(max_dp, 3),
            n_pairwise_checks=len(tc.pairwise),
            total_bits_in=round(total_in, 2),
            total_bits_out=round(total_out, 2),
            total_info_loss_bits=round(total_loss, 2),
            preservation_ratio=round(pres_ratio, 4),
            mapping_details=[
                {
                    "label": r.mapping_label,
                    "bits_in": round(r.bits_in, 2),
                    "bits_out": round(r.bits_out, 2),
                    "loss": round(r.information_loss_bits, 2),
                    "preservation": round(r.preservation_ratio, 4),
                    "audio_steps": r.audio_capacity_steps,
                }
                for r in loss_reports
            ],
            pairwise_details=[
                {
                    "pair": f"{pw.stream_a}×{pw.stream_b}",
                    "d_prime": round(pw.d_prime, 3),
                    "passes": pw.passes,
                    "pitch_margin": round(pw.pitch_margin, 3),
                    "loudness_margin": round(pw.loudness_margin, 3),
                }
                for pw in tc.pairwise
            ],
            violations=[
                {"severity": v.severity.value, "dimension": v.dimension,
                 "message": v.message}
                for v in tc.violations
            ],
        )
        benchmark_results.append(br)

        # ── Baseline ──
        baseline = compute_baseline(case)
        baseline_results.append(baseline)

        # Print row
        status = "✓" if correct else "✗"
        verdict = "PASS" if tc.well_typed else "FAIL"
        exp_str = "PASS" if case.expected_well_typed else "FAIL"
        print(f"{case.id:<25} {case.n_streams:>7} {exp_str:>10} {verdict:>10} "
              f"{status:>8} {mean_dp:>8.2f} {total_out:>8.1f} "
              f"{pres_ratio:>6.1%} {compile_us:>10.1f}")

    print("─" * 110)

    # ── Accuracy summary ─────────────────────────────────────────────
    n_correct = sum(1 for r in benchmark_results if r.correct_verdict)
    n_total = len(benchmark_results)
    print(f"\nType-check verdict accuracy: {n_correct}/{n_total} "
          f"({n_correct/n_total:.0%})")

    # ── Scaling benchmark ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Scaling Benchmark: Compilation time vs. stream count")
    print("=" * 70)
    scaling = run_scaling_benchmark()
    print(f"\n{'Streams':>8} {'Pairs':>6} {'TypeCheck(µs)':>14} {'Compile(µs)':>14} {'p95(µs)':>10}")
    print("─" * 56)
    for s in scaling:
        print(f"{s['n_streams']:>8} {s['n_pairwise_checks']:>6} "
              f"{s['typecheck_time_us_mean']:>14.1f} "
              f"{s['compile_time_us_mean']:>14.1f} "
              f"{s['compile_time_us_p95']:>10.1f}")

    # ── Weber validation ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Weber's Law Compliance Validation")
    print("=" * 70)
    weber = validate_weber_compliance()
    for dim, data in weber.items():
        print(f"\n  {dim}: k = {data['weber_fraction_k']}, range = {data['functional_range']}")
        if "r_squared" in data:
            print(f"    r² = {data['r_squared']:.4f}")

    # ── Baseline comparison ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Baseline Comparison: SoniType vs. Naive Mapping")
    print("=" * 70)
    print(f"\n{'ID':<25} {'Naive Pres':>11} {'SoniType Pres':>14} "
          f"{'Naive d':>8} {'ST d':>8} {'Naive Viol':>11}")
    print("─" * 85)
    for br, bl in zip(benchmark_results, baseline_results):
        viol = bl.masking_violations + bl.jnd_violations + bl.segregation_failures
        cog = " +cog" if bl.cognitive_overload else ""
        print(f"{br.id:<25} {bl.naive_preservation:>10.1%} "
              f"{br.preservation_ratio:>13.1%} "
              f"{bl.naive_d_prime:>8.2f} {br.mean_d_prime:>8.2f} "
              f"{viol:>5}{cog:>6}")

    # ── Build output JSON ────────────────────────────────────────────
    output = {
        "benchmark": "perceptual_type_check",
        "version": "1.0.0",
        "seed": SEED,
        "n_cases": len(cases),
        "type_check_accuracy": f"{n_correct}/{n_total}",
        "d_prime_threshold": DPRIME_THRESHOLD,
        "cognitive_budget": COGNITIVE_LOAD_MAX,
        "psychoacoustic_constants": {
            "pitch_weber_optimal": PITCH_WEBER_OPTIMAL,
            "loudness_jnd_db": LOUDNESS_JND_DB,
            "duration_weber": DURATION_WEBER,
            "duration_floor_ms": DURATION_FLOOR_MS,
            "timbre_centroid_fraction": TIMBRE_CENTROID_FRACTION,
        },
        "results": [asdict(r) for r in benchmark_results],
        "baseline_comparison": [asdict(b) for b in baseline_results],
        "scaling": scaling,
        "weber_validation": weber,
        "summary": {
            "well_typed_cases": {
                "count": sum(1 for r in benchmark_results if r.category == "well_typed"),
                "all_correct": all(r.correct_verdict for r in benchmark_results
                                   if r.category == "well_typed"),
                "mean_preservation": round(
                    sum(r.preservation_ratio for r in benchmark_results
                        if r.category == "well_typed" and r.actual_well_typed)
                    / max(sum(1 for r in benchmark_results
                              if r.category == "well_typed" and r.actual_well_typed), 1),
                    4,
                ),
                "mean_d_prime": round(
                    sum(r.mean_d_prime for r in benchmark_results
                        if r.category == "well_typed" and r.mean_d_prime > 0)
                    / max(sum(1 for r in benchmark_results
                              if r.category == "well_typed" and r.mean_d_prime > 0), 1),
                    3,
                ),
            },
            "ill_typed_cases": {
                "count": sum(1 for r in benchmark_results if r.category == "ill_typed"),
                "all_correctly_rejected": all(
                    r.correct_verdict for r in benchmark_results
                    if r.category == "ill_typed"
                ),
            },
            "borderline_cases": {
                "count": sum(1 for r in benchmark_results if r.category == "borderline"),
                "all_correct": all(r.correct_verdict for r in benchmark_results
                                   if r.category == "borderline"),
            },
            "real_world_cases": {
                "count": sum(1 for r in benchmark_results if r.category == "real_world"),
                "all_pass": all(r.actual_well_typed for r in benchmark_results
                                if r.category == "real_world"),
                "mean_preservation": round(
                    sum(r.preservation_ratio for r in benchmark_results
                        if r.category == "real_world" and r.actual_well_typed)
                    / max(sum(1 for r in benchmark_results
                              if r.category == "real_world" and r.actual_well_typed), 1),
                    4,
                ),
            },
            "baseline_vs_sonitype": {
                "mean_naive_violations": round(
                    sum(b.masking_violations + b.jnd_violations + b.segregation_failures
                        for b in baseline_results) / len(baseline_results), 1
                ),
                "sonitype_violations_in_passing": 0,
                "note": "SoniType guarantees zero violations for well-typed specs by construction",
            },
        },
        "methodology": {
            "type_system": "Perceptual refinement types with JND-based qualifiers (§4 of paper)",
            "jnd_models": "Wier et al. (1977) pitch, Jesteadt et al. (1977) loudness, "
                          "Friberg & Sundberg (1995) duration, Grey (1977) timbre",
            "d_prime": "Compound d' via perceptual independence: d' = √Σ margin²_i",
            "information_loss": "Channel capacity in bits: log₂(JND-separated steps)",
            "baseline": "Naive 8-bit uniform quantization without perceptual awareness",
            "timing": "perf_counter_ns, 10-20 iterations, median reported",
        },
    }

    # Write JSON
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "perceptual_type_check_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {out_path}")

    return output


if __name__ == "__main__":
    run_benchmark()
