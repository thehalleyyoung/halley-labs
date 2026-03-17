#!/usr/bin/env python3
"""
Perceptual Type Checker for the SoniType sonification compiler.

Implements the core perceptual type system described in Section 4 of the paper:
  - Perceptual types: Pitch, Duration, Timbre, Loudness with psychoacoustic bounds
  - JND-based information-preservation checking
  - Non-local pairwise constraint validation (masking, segregation)
  - Cognitive load budget enforcement
  - Sonification compilation from typed specs to synthesis instructions
  - Perceptual information-loss estimation

This module grounds the formal typing rules (T-Stream, T-Map, T-Compose) from
Appendix A with executable semantics, using the same Weber-fraction JND models
and Schroeder spreading function implemented in the Rust crate
sonitype-psychoacoustic/src/jnd.rs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple


# ── Psychoacoustic constants ──────────────────────────────────────────────

# Wier et al. (1977) Weber fraction in the 500–5000 Hz optimal region.
PITCH_WEBER_OPTIMAL = 0.0035

# Jesteadt et al. (1977): ~1 dB at moderate sensation levels.
LOUDNESS_JND_DB = 1.0

# Duration JND: 10 % Weber fraction with 10 ms floor (Friberg & Sundberg 1995).
DURATION_WEBER = 0.10
DURATION_FLOOR_MS = 10.0

# Timbre: spectral centroid JND ≈ 5 % (Grey 1977, McAdams et al. 1995).
TIMBRE_CENTROID_FRACTION = 0.05

# Cowan (2001) working-memory capacity.
COGNITIVE_LOAD_MAX = 5.0

# Default d' threshold for type-check pass (≈ 93 % 2AFC accuracy).
DPRIME_THRESHOLD = 1.5


# ── Unit conversion helpers (mirrors sonitype-core/src/units.rs) ──────────

def hz_to_bark(freq_hz: float) -> float:
    """Zwicker formula: bark = 13·arctan(0.00076·f) + 3.5·arctan((f/7500)²)."""
    return 13.0 * math.atan(0.00076 * freq_hz) + 3.5 * math.atan((freq_hz / 7500.0) ** 2)


def hz_to_cents(f1: float, f2: float) -> float:
    """Interval between two frequencies in cents."""
    if f1 <= 0 or f2 <= 0:
        return 0.0
    return 1200.0 * math.log2(f2 / f1)


# ── Scale Enum ────────────────────────────────────────────────────────────

class Scale(Enum):
    """Musical scale constraint for pitch mappings."""
    CHROMATIC = auto()
    MAJOR = auto()
    MINOR = auto()
    PENTATONIC = auto()
    WHOLE_TONE = auto()
    FREE = auto()  # unconstrained continuous


# ── Perceptual Types ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class PitchType:
    """Pitch ∈ [lo, hi] Hz on a given scale.

    Type-checks iff the range spans at least one pitch JND, ensuring the
    mapping can convey at least a binary distinction.
    """
    lo_hz: float
    hi_hz: float
    scale: Scale = Scale.FREE

    def range_hz(self) -> float:
        return self.hi_hz - self.lo_hz

    def jnd_at(self, freq_hz: float) -> float:
        """Frequency-dependent JND (Hz).  Mirrors PitchJnd::jnd_frequency."""
        f = max(abs(freq_hz), 20.0)
        if f < 500.0:
            wf = PITCH_WEBER_OPTIMAL * math.sqrt(500.0 / f)
        elif f > 5000.0:
            wf = PITCH_WEBER_OPTIMAL * math.sqrt(f / 5000.0)
        else:
            wf = PITCH_WEBER_OPTIMAL
        return f * wf

    def distinguishable_steps(self) -> int:
        """Count how many JND-separated steps fit in [lo, hi]."""
        f = self.lo_hz
        steps = 0
        while f < self.hi_hz:
            f += self.jnd_at(f)
            steps += 1
        return steps


@dataclass(frozen=True)
class DurationType:
    """Duration ∈ [min_ms, max_ms].

    Type-checks iff the range spans at least one temporal JND.
    """
    min_ms: float
    max_ms: float

    def range_ms(self) -> float:
        return self.max_ms - self.min_ms

    def jnd_at(self, dur_ms: float) -> float:
        """Duration JND: max(10 % · dur, 10 ms)."""
        return max(DURATION_WEBER * dur_ms, DURATION_FLOOR_MS)

    def distinguishable_steps(self) -> int:
        d = self.min_ms
        steps = 0
        while d < self.max_ms:
            d += self.jnd_at(d)
            steps += 1
        return steps


@dataclass(frozen=True)
class TimbreType:
    """Timbre constrained by spectral-centroid range [lo, hi] Hz.

    The centroid JND is ≈ 5 % of the centroid value.
    """
    centroid_lo_hz: float
    centroid_hi_hz: float

    def range_hz(self) -> float:
        return self.centroid_hi_hz - self.centroid_lo_hz

    def jnd_at(self, centroid_hz: float) -> float:
        return TIMBRE_CENTROID_FRACTION * max(abs(centroid_hz), 1.0)

    def distinguishable_steps(self) -> int:
        c = self.centroid_lo_hz
        steps = 0
        while c < self.centroid_hi_hz:
            c += self.jnd_at(c)
            steps += 1
        return steps


@dataclass(frozen=True)
class LoudnessType:
    """Loudness ∈ [lo_phon, hi_phon].

    JND ≈ 1 dB at moderate levels (Jesteadt et al. 1977).
    """
    lo_phon: float
    hi_phon: float

    def range_phon(self) -> float:
        return self.hi_phon - self.lo_phon

    def jnd_at(self, level_phon: float) -> float:
        if level_phon < 20.0:
            return LOUDNESS_JND_DB + 0.5 * (20.0 - level_phon) / 20.0
        elif level_phon > 80.0:
            return max(LOUDNESS_JND_DB - 0.1 * (level_phon - 80.0) / 40.0, 0.5)
        return LOUDNESS_JND_DB

    def distinguishable_steps(self) -> int:
        lev = self.lo_phon
        steps = 0
        while lev < self.hi_phon:
            lev += self.jnd_at(lev)
            steps += 1
        return steps


# ── Data-dimension descriptor ─────────────────────────────────────────────

@dataclass
class DataDimension:
    """One column of input data with distribution summary."""
    name: str
    min_val: float
    max_val: float
    cardinality: Optional[int] = None  # for discrete/categorical

    def range(self) -> float:
        return self.max_val - self.min_val


# ── Mapping: data dimension → audio dimension ─────────────────────────────

@dataclass
class DimensionMapping:
    """Maps a data dimension to an audio perceptual type."""
    data_dim: DataDimension
    audio_type: PitchType | DurationType | TimbreType | LoudnessType
    label: str = ""

    def __post_init__(self):
        if not self.label:
            atype = type(self.audio_type).__name__.replace("Type", "")
            self.label = f"{self.data_dim.name}→{atype}"


# ── Stream specification ──────────────────────────────────────────────────

@dataclass
class StreamSpec:
    """A single sonification stream: one or more dimension mappings."""
    name: str
    mappings: List[DimensionMapping]
    base_frequency_hz: float = 440.0
    level_db: float = 60.0
    onset_ms: float = 0.0
    cognitive_weight: float = 1.0


# ── Sonification specification ────────────────────────────────────────────

@dataclass
class SonificationSpec:
    """Top-level specification: multiple concurrent streams."""
    streams: List[StreamSpec]
    cognitive_budget: float = COGNITIVE_LOAD_MAX


# ── Type-check result types ───────────────────────────────────────────────

class Severity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class TypeViolation:
    """A single type-system violation."""
    severity: Severity
    dimension: str
    message: str
    jnd_required: Optional[float] = None
    jnd_available: Optional[float] = None


@dataclass
class InformationLossReport:
    """Per-mapping information-loss estimate."""
    mapping_label: str
    data_cardinality: float      # distinct values or continuous range / resolution
    audio_capacity_steps: int    # JND-separated steps in the audio range
    bits_in: float               # log2 of data cardinality
    bits_out: float              # log2 of audio capacity
    information_loss_bits: float  # max(0, bits_in − bits_out)
    preservation_ratio: float    # bits_out / bits_in  (clamped to [0,1])


@dataclass
class PairwiseCheck:
    """Result of checking two streams against each other."""
    stream_a: str
    stream_b: str
    d_prime: float
    passes: bool
    pitch_margin: float
    loudness_margin: float
    detail: str = ""


@dataclass
class TypeCheckResult:
    """Aggregate result of the perceptual type checker."""
    well_typed: bool
    violations: List[TypeViolation]
    pairwise: List[PairwiseCheck]
    information_loss: List[InformationLossReport]
    cognitive_load_total: float
    cognitive_budget: float

    def summary(self) -> str:
        status = "PASS" if self.well_typed else "FAIL"
        nerr = sum(1 for v in self.violations if v.severity == Severity.ERROR)
        nwarn = sum(1 for v in self.violations if v.severity == Severity.WARNING)
        return (
            f"TypeCheck {status}: {nerr} errors, {nwarn} warnings, "
            f"cognitive load {self.cognitive_load_total:.1f}/{self.cognitive_budget:.1f}"
        )


# ── Perceptual Type Checker ───────────────────────────────────────────────

class PerceptualTypeChecker:
    """Core type-checking engine.

    Implements the typing rules from §4 and Appendix A:
      T-Stream: each mapping satisfies per-dimension JND viability
      T-Compose: pairwise d' ≥ τ_d and cognitive load within budget
    """

    def __init__(self, d_prime_threshold: float = DPRIME_THRESHOLD):
        self.d_prime_threshold = d_prime_threshold

    # ── per-mapping checks (T-Stream / T-Map) ────────────────────────

    def _check_mapping(self, m: DimensionMapping) -> Tuple[List[TypeViolation], InformationLossReport]:
        violations: List[TypeViolation] = []
        at = m.audio_type
        steps = at.distinguishable_steps()

        if steps < 1:
            violations.append(TypeViolation(
                severity=Severity.ERROR,
                dimension=m.label,
                message=f"Audio range collapses below 1 JND ({steps} steps); "
                        f"mapping conveys no information",
            ))

        # information-loss estimate
        if m.data_dim.cardinality is not None:
            data_card = float(m.data_dim.cardinality)
        else:
            # continuous: approximate by range / smallest JND in range
            if isinstance(at, PitchType):
                data_card = max(m.data_dim.range() / at.jnd_at(at.lo_hz), 1)
            elif isinstance(at, DurationType):
                data_card = max(m.data_dim.range() / at.jnd_at(at.min_ms), 1)
            elif isinstance(at, TimbreType):
                data_card = max(m.data_dim.range() / at.jnd_at(at.centroid_lo_hz), 1)
            elif isinstance(at, LoudnessType):
                data_card = max(m.data_dim.range() / at.jnd_at(at.lo_phon), 1)
            else:
                data_card = max(m.data_dim.range(), 1)

        bits_in = math.log2(max(data_card, 1))
        bits_out = math.log2(max(steps, 1))
        loss = max(bits_in - bits_out, 0.0)
        ratio = min(bits_out / max(bits_in, 1e-9), 1.0)

        if loss > 1.0:
            violations.append(TypeViolation(
                severity=Severity.WARNING,
                dimension=m.label,
                message=f"Information loss ≈ {loss:.1f} bits "
                        f"({steps} audio steps for {data_card:.0f} data levels)",
            ))

        report = InformationLossReport(
            mapping_label=m.label,
            data_cardinality=data_card,
            audio_capacity_steps=steps,
            bits_in=bits_in,
            bits_out=bits_out,
            information_loss_bits=loss,
            preservation_ratio=ratio,
        )
        return violations, report

    # ── pairwise d' check (T-Compose) ────────────────────────────────

    def _pairwise_dprime(self, sa: StreamSpec, sb: StreamSpec) -> PairwiseCheck:
        """Compute compound d' between two streams (Eq. 6 in paper)."""
        # pitch margin
        fa, fb = sa.base_frequency_hz, sb.base_frequency_hz
        pitch_jnd = max(fa, fb, 20.0) * PITCH_WEBER_OPTIMAL
        pitch_margin = abs(fa - fb) / max(pitch_jnd, 1e-9)

        # loudness margin
        la, lb = sa.level_db, sb.level_db
        loud_jnd = LOUDNESS_JND_DB
        loud_margin = abs(la - lb) / max(loud_jnd, 1e-9)

        # compound d' (perceptual independence assumption)
        d_prime = math.sqrt(pitch_margin ** 2 + loud_margin ** 2)
        passes = d_prime >= self.d_prime_threshold

        return PairwiseCheck(
            stream_a=sa.name,
            stream_b=sb.name,
            d_prime=d_prime,
            passes=passes,
            pitch_margin=pitch_margin,
            loudness_margin=loud_margin,
            detail=f"d'={d_prime:.2f} (threshold={self.d_prime_threshold})",
        )

    # ── main entry point ─────────────────────────────────────────────

    def check(self, spec: SonificationSpec) -> TypeCheckResult:
        """Type-check a full sonification specification."""
        violations: List[TypeViolation] = []
        loss_reports: List[InformationLossReport] = []
        pairwise: List[PairwiseCheck] = []

        # 1. Per-stream / per-mapping checks (T-Stream)
        for stream in spec.streams:
            for mapping in stream.mappings:
                mviol, mloss = self._check_mapping(mapping)
                violations.extend(mviol)
                loss_reports.append(mloss)

        # 2. Pairwise stream checks (T-Compose: masking + segregation)
        for i, sa in enumerate(spec.streams):
            for sb in spec.streams[i + 1:]:
                pw = self._pairwise_dprime(sa, sb)
                pairwise.append(pw)
                if not pw.passes:
                    violations.append(TypeViolation(
                        severity=Severity.ERROR,
                        dimension=f"{sa.name}×{sb.name}",
                        message=f"Streams indiscriminable: d'={pw.d_prime:.2f} "
                                f"< threshold {self.d_prime_threshold}",
                    ))

        # 3. Cognitive load check (φ_cog)
        total_load = sum(s.cognitive_weight for s in spec.streams)
        if total_load > spec.cognitive_budget:
            violations.append(TypeViolation(
                severity=Severity.ERROR,
                dimension="cognitive_load",
                message=f"Cognitive load {total_load:.1f} exceeds budget "
                        f"{spec.cognitive_budget:.1f}",
            ))

        well_typed = not any(v.severity == Severity.ERROR for v in violations)

        return TypeCheckResult(
            well_typed=well_typed,
            violations=violations,
            pairwise=pairwise,
            information_loss=loss_reports,
            cognitive_load_total=total_load,
            cognitive_budget=spec.cognitive_budget,
        )


# ── Sonification Compiler ────────────────────────────────────────────────

@dataclass
class SynthesisInstruction:
    """A single audio-synthesis instruction emitted by the compiler."""
    stream_name: str
    parameter: str
    mapping_function: str  # human-readable description
    range_lo: float
    range_hi: float
    jnd_step: float
    quantised_levels: int


@dataclass
class CompilationResult:
    """Output of the SonificationCompiler."""
    success: bool
    type_check: TypeCheckResult
    instructions: List[SynthesisInstruction]
    warnings: List[str]


class SonificationCompiler:
    """Compile a typed SonificationSpec into synthesis instructions.

    Validates type constraints, generates mapping functions that respect
    JND bounds, and reports information loss for any dimension that
    collapses below the JND threshold.
    """

    def __init__(self, d_prime_threshold: float = DPRIME_THRESHOLD):
        self.checker = PerceptualTypeChecker(d_prime_threshold)

    def compile(self, spec: SonificationSpec) -> CompilationResult:
        tc = self.checker.check(spec)
        instructions: List[SynthesisInstruction] = []
        warnings: List[str] = []

        if not tc.well_typed:
            return CompilationResult(
                success=False,
                type_check=tc,
                instructions=[],
                warnings=[f"Type errors prevent compilation: {tc.summary()}"],
            )

        for stream in spec.streams:
            for mapping in stream.mappings:
                at = mapping.audio_type
                steps = at.distinguishable_steps()

                if isinstance(at, PitchType):
                    param = "pitch"
                    lo, hi = at.lo_hz, at.hi_hz
                    jnd = at.jnd_at(at.lo_hz)
                    func = f"linear_map({mapping.data_dim.name}, [{lo:.1f}, {hi:.1f}] Hz)"
                elif isinstance(at, DurationType):
                    param = "duration"
                    lo, hi = at.min_ms, at.max_ms
                    jnd = at.jnd_at(at.min_ms)
                    func = f"linear_map({mapping.data_dim.name}, [{lo:.1f}, {hi:.1f}] ms)"
                elif isinstance(at, TimbreType):
                    param = "timbre_centroid"
                    lo, hi = at.centroid_lo_hz, at.centroid_hi_hz
                    jnd = at.jnd_at(at.centroid_lo_hz)
                    func = f"linear_map({mapping.data_dim.name}, centroid [{lo:.0f}, {hi:.0f}] Hz)"
                elif isinstance(at, LoudnessType):
                    param = "loudness"
                    lo, hi = at.lo_phon, at.hi_phon
                    jnd = at.jnd_at(at.lo_phon)
                    func = f"linear_map({mapping.data_dim.name}, [{lo:.1f}, {hi:.1f}] phon)"
                else:
                    continue

                instructions.append(SynthesisInstruction(
                    stream_name=stream.name,
                    parameter=param,
                    mapping_function=func,
                    range_lo=lo,
                    range_hi=hi,
                    jnd_step=jnd,
                    quantised_levels=steps,
                ))

                if steps < 8:
                    warnings.append(
                        f"{mapping.label}: only {steps} distinguishable levels "
                        f"(< 3 bits); consider widening the audio range"
                    )

        return CompilationResult(
            success=True,
            type_check=tc,
            instructions=instructions,
            warnings=warnings,
        )


# ── Perceptual Loss Estimator ────────────────────────────────────────────

class PerceptualLossEstimator:
    """Quantify information loss from a data→audio mapping.

    Computes channel capacity in bits for each audio dimension, compares
    against the data entropy, and returns per-dimension and aggregate loss.
    """

    @staticmethod
    def estimate(spec: SonificationSpec) -> List[InformationLossReport]:
        checker = PerceptualTypeChecker()
        tc = checker.check(spec)
        return tc.information_loss

    @staticmethod
    def aggregate_loss(reports: List[InformationLossReport]) -> Dict[str, float]:
        total_in = sum(r.bits_in for r in reports)
        total_out = sum(r.bits_out for r in reports)
        total_loss = sum(r.information_loss_bits for r in reports)
        return {
            "total_bits_in": total_in,
            "total_bits_out": total_out,
            "total_loss_bits": total_loss,
            "aggregate_preservation_ratio": total_out / max(total_in, 1e-9),
        }


# ── Self-test / demo ─────────────────────────────────────────────────────

def _demo() -> None:
    """Run an end-to-end demo that exercises all three components."""

    # Data schema: 3 dimensions
    d_price = DataDimension("price", min_val=0.0, max_val=200.0)
    d_volume = DataDimension("volume", min_val=0.0, max_val=1e6)
    d_volatility = DataDimension("volatility", min_val=0.0, max_val=100.0)

    # Well-typed spec: 2 streams, well-separated
    spec_ok = SonificationSpec(streams=[
        StreamSpec(
            name="price_stream",
            mappings=[
                DimensionMapping(d_price, PitchType(200.0, 2000.0)),
                DimensionMapping(d_volatility, LoudnessType(40.0, 80.0)),
            ],
            base_frequency_hz=440.0,
            level_db=60.0,
        ),
        StreamSpec(
            name="volume_stream",
            mappings=[
                DimensionMapping(d_volume, DurationType(100.0, 1000.0)),
            ],
            base_frequency_hz=1200.0,
            level_db=55.0,
        ),
    ])

    # Compile
    compiler = SonificationCompiler()
    result = compiler.compile(spec_ok)
    print("=== Well-typed specification ===")
    print(result.type_check.summary())
    for instr in result.instructions:
        print(f"  {instr.stream_name}.{instr.parameter}: {instr.mapping_function} "
              f"({instr.quantised_levels} levels, JND={instr.jnd_step:.2f})")
    for w in result.warnings:
        print(f"  ⚠ {w}")
    print()

    # Ill-typed spec: cognitive overload + near-identical streams
    spec_bad = SonificationSpec(streams=[
        StreamSpec(name=f"s{i}", mappings=[
            DimensionMapping(d_price, PitchType(440.0, 441.0)),
        ], base_frequency_hz=440.0 + i * 0.1)
        for i in range(7)
    ])

    result_bad = compiler.compile(spec_bad)
    print("=== Ill-typed specification ===")
    print(result_bad.type_check.summary())
    for v in result_bad.type_check.violations:
        print(f"  [{v.severity.value}] {v.dimension}: {v.message}")
    print()

    # Information-loss estimate
    reports = PerceptualLossEstimator.estimate(spec_ok)
    agg = PerceptualLossEstimator.aggregate_loss(reports)
    print("=== Information Loss ===")
    for r in reports:
        print(f"  {r.mapping_label}: {r.bits_in:.1f} → {r.bits_out:.1f} bits "
              f"(loss={r.information_loss_bits:.1f}, "
              f"preservation={r.preservation_ratio:.0%})")
    print(f"  Aggregate: {agg['total_bits_in']:.1f} → {agg['total_bits_out']:.1f} bits "
          f"(preservation={agg['aggregate_preservation_ratio']:.0%})")


if __name__ == "__main__":
    _demo()
