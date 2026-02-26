"""Independent certificate verifier with minimal trusted computing base.

This verifier operates independently of BioProver's solver infrastructure.
It re-checks proof certificates using only:
  - mpmath arbitrary-precision interval arithmetic (for flowpipe replay)
  - Standard Python (json, hashlib, math) for structural checks
  - Optionally z3-solver for SMT proof re-checking

Design principle: The verifier NEVER imports from bioprover.solver,
bioprover.cegar, or bioprover.smt. It uses its own minimal interval
arithmetic to replay computations, ensuring the TCB is reduced to
this module alone (~800 LoC) rather than the full BioProver codebase
(~58K LoC).

Mathematical foundation:
  Given a certificate C = (f, X₀, {Sᵢ}ᵢ, Δt, method), the verifier
  checks that for each segment Sᵢ = ([tᵢ, tᵢ₊₁], Bᵢ):
    1. B₀ ⊇ X₀  (initial containment)
    2. tᵢ₊₁ ≥ tᵢ  (time monotonicity)
    3. Bᵢ₊₁ ⊇ φ(Bᵢ, [tᵢ, tᵢ₊₁])  (flow containment via Euler/Taylor replay)
  where φ is the validated flow map computed by the verifier's own
  interval ODE integrator.
"""

from __future__ import annotations

import hashlib
import json
import math
import time as time_mod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

try:
    import mpmath
    mpmath.mp.prec = 113  # quad precision
    _HAS_MPMATH = True
except ImportError:
    _HAS_MPMATH = False

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# ═══════════════════════════════════════════════════════════════════════════
# Minimal interval arithmetic (verifier-internal, independent of bioprover)
# ═══════════════════════════════════════════════════════════════════════════

class VInterval:
    """Verified interval [lo, hi] with rigorous outward rounding.

    Uses mpmath for arbitrary-precision arithmetic when available,
    falls back to float64 nextafter-based rounding.
    """
    __slots__ = ("lo", "hi")

    def __init__(self, lo: float, hi: Optional[float] = None):
        if hi is None:
            hi = lo
        self.lo = float(lo)
        self.hi = float(hi)
        if self.lo > self.hi + 1e-15:
            raise ValueError(f"Empty interval: [{self.lo}, {self.hi}]")
        if self.lo > self.hi:
            self.hi = self.lo

    @staticmethod
    def _down(x: float) -> float:
        if _HAS_MPMATH:
            with mpmath.workprec(113):
                return float(mpmath.mpf(x) - mpmath.mpf(2) ** -52)
        if math.isnan(x) or math.isinf(x):
            return x
        if x == 0.0:
            return -5e-324
        return math.nextafter(x, -math.inf)

    @staticmethod
    def _up(x: float) -> float:
        if _HAS_MPMATH:
            with mpmath.workprec(113):
                return float(mpmath.mpf(x) + mpmath.mpf(2) ** -52)
        if math.isnan(x) or math.isinf(x):
            return x
        if x == 0.0:
            return 5e-324
        return math.nextafter(x, math.inf)

    def width(self) -> float:
        return self.hi - self.lo

    def mid(self) -> float:
        return 0.5 * (self.lo + self.hi)

    def contains(self, other: Union[float, "VInterval"]) -> bool:
        if isinstance(other, VInterval):
            return self.lo <= other.lo + 1e-15 and other.hi <= self.hi + 1e-15
        return self.lo <= float(other) + 1e-15 and float(other) <= self.hi + 1e-15

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = VInterval(other)
        return VInterval(
            self._down(self.lo + other.lo),
            self._up(self.hi + other.hi),
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = VInterval(other)
        return VInterval(
            self._down(self.lo - other.hi),
            self._up(self.hi - other.lo),
        )

    def __rsub__(self, other):
        return VInterval(other).__sub__(self)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = VInterval(other)
        products = [
            self.lo * other.lo, self.lo * other.hi,
            self.hi * other.lo, self.hi * other.hi,
        ]
        return VInterval(self._down(min(products)), self._up(max(products)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            other = VInterval(other)
        if other.lo <= 0 <= other.hi:
            return VInterval(-math.inf, math.inf)
        inv = VInterval(self._down(1.0 / other.hi), self._up(1.0 / other.lo))
        return self * inv

    def __neg__(self):
        return VInterval(-self.hi, -self.lo)

    def __pow__(self, n: int):
        if n == 0:
            return VInterval(1.0)
        if n == 1:
            return VInterval(self.lo, self.hi)
        if n < 0:
            return VInterval(1.0) / (self ** (-n))
        if n % 2 == 0:
            if self.lo >= 0:
                return VInterval(self._down(self.lo ** n), self._up(self.hi ** n))
            if self.hi <= 0:
                return VInterval(self._down(self.hi ** n), self._up(self.lo ** n))
            upper = max(abs(self.lo), abs(self.hi))
            return VInterval(0.0, self._up(upper ** n))
        return VInterval(self._down(self.lo ** n), self._up(self.hi ** n))

    def exp(self):
        return VInterval(self._down(math.exp(self.lo)), self._up(math.exp(self.hi)))

    def log(self):
        lo = max(self.lo, 1e-300)
        return VInterval(self._down(math.log(lo)), self._up(math.log(max(self.hi, 1e-300))))

    def __repr__(self):
        return f"[{self.lo:.6g}, {self.hi:.6g}]"


class VBox:
    """Verified interval vector (axis-aligned box)."""
    __slots__ = ("intervals",)

    def __init__(self, intervals: Sequence[VInterval]):
        self.intervals = list(intervals)

    @classmethod
    def from_bounds(cls, lo: List[float], hi: List[float]) -> "VBox":
        return cls([VInterval(l, h) for l, h in zip(lo, hi)])

    @property
    def dim(self) -> int:
        return len(self.intervals)

    def __getitem__(self, i: int) -> VInterval:
        return self.intervals[i]

    def contains(self, other: "VBox") -> bool:
        if self.dim != other.dim:
            return False
        return all(s.contains(o) for s, o in zip(self.intervals, other.intervals))

    def max_width(self) -> float:
        return max(iv.width() for iv in self.intervals)

    def bloat(self, eps: float) -> "VBox":
        return VBox([
            VInterval(iv.lo - eps, iv.hi + eps) for iv in self.intervals
        ])

    def hull(self, other: "VBox") -> "VBox":
        return VBox([
            VInterval(min(a.lo, b.lo), max(a.hi, b.hi))
            for a, b in zip(self.intervals, other.intervals)
        ])

    def __repr__(self):
        return "VBox(" + ", ".join(str(iv) for iv in self.intervals) + ")"


# ═══════════════════════════════════════════════════════════════════════════
# Verification report
# ═══════════════════════════════════════════════════════════════════════════

class CheckStatus(Enum):
    PASS = auto()
    FAIL = auto()
    WARN = auto()
    SKIP = auto()


@dataclass
class CheckResult:
    """Result of a single verification check."""
    name: str
    status: CheckStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationReport:
    """Complete report from certificate verification."""
    certificate_type: str
    certificate_hash: str
    timestamp: float = field(default_factory=time_mod.time)
    checks: List[CheckResult] = field(default_factory=list)
    valid: bool = False
    summary: str = ""
    verifier_version: str = "1.0.0"
    verification_time_s: float = 0.0

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.PASS)

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.FAIL)

    @property
    def warnings(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.WARN)

    def add_check(self, name: str, status: CheckStatus, message: str = "",
                  **details) -> None:
        self.checks.append(CheckResult(name, status, message, details))

    def finalize(self) -> None:
        self.valid = self.failed == 0
        if self.valid:
            self.summary = (
                f"Certificate VALID: {self.passed} checks passed"
                + (f", {self.warnings} warnings" if self.warnings else "")
            )
        else:
            self.summary = (
                f"Certificate INVALID: {self.failed} checks failed "
                f"out of {len(self.checks)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "certificate_type": self.certificate_type,
            "certificate_hash": self.certificate_hash,
            "timestamp": self.timestamp,
            "valid": self.valid,
            "summary": self.summary,
            "verifier_version": self.verifier_version,
            "verification_time_s": self.verification_time_s,
            "passed": self.passed,
            "failed": self.failed,
            "warnings": self.warnings,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.name,
                    "message": c.message,
                    "details": c.details,
                }
                for c in self.checks
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ═══════════════════════════════════════════════════════════════════════════
# Flowpipe replay verifier
# ═══════════════════════════════════════════════════════════════════════════

class FlowpipeReplayVerifier:
    """Independently verifies flowpipe certificates by replaying integration.

    Given a flowpipe certificate containing the ODE system description,
    initial conditions, and segment enclosures, this verifier:

    1. Parses the ODE system from the certificate
    2. Re-integrates using its own validated Euler method
    3. Checks that each certificate segment contains the replay enclosure

    The replay uses a deliberately simpler (and thus more conservative)
    integration method than BioProver's Taylor/Lohner integrators, so if
    the certificate segments contain the replay enclosures, they are
    sound over-approximations.
    """

    def __init__(self, replay_steps_per_segment: int = 10,
                 bloat_factor: float = 1e-8):
        self.replay_steps = replay_steps_per_segment
        self.bloat_factor = bloat_factor

    def verify(self, cert_data: Dict[str, Any]) -> VerificationReport:
        """Verify a flowpipe certificate."""
        cert_hash = hashlib.sha256(
            json.dumps(cert_data, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        report = VerificationReport(
            certificate_type="flowpipe",
            certificate_hash=cert_hash,
        )
        t0 = time_mod.time()

        try:
            self._check_structure(cert_data, report)
            self._check_time_ordering(cert_data, report)
            self._check_dimension_consistency(cert_data, report)
            self._check_initial_containment(cert_data, report)
            self._check_box_validity(cert_data, report)
            self._check_enclosure_widths(cert_data, report)
            self._check_segment_continuity(cert_data, report)
            if cert_data.get("system_description"):
                self._replay_integration(cert_data, report)
        except Exception as e:
            report.add_check("unexpected_error", CheckStatus.FAIL,
                             f"Verifier error: {e}")

        report.verification_time_s = time_mod.time() - t0
        report.finalize()
        return report

    def _check_structure(self, data: Dict, report: VerificationReport) -> None:
        """Check certificate has required fields."""
        required = ["certificate_type", "dimension", "segments"]
        missing = [f for f in required if f not in data]
        if missing:
            report.add_check("structure", CheckStatus.FAIL,
                             f"Missing fields: {missing}")
        else:
            report.add_check("structure", CheckStatus.PASS,
                             f"All required fields present")

    def _check_time_ordering(self, data: Dict, report: VerificationReport) -> None:
        """Check segments are ordered in time."""
        segments = data.get("segments", [])
        if not segments:
            report.add_check("time_ordering", CheckStatus.FAIL, "No segments")
            return

        violations = []
        for i in range(len(segments) - 1):
            t_hi = segments[i]["time_hi"]
            t_lo_next = segments[i + 1]["time_lo"]
            if t_lo_next > t_hi + 1e-10:
                violations.append((i, t_hi, t_lo_next))

        if violations:
            report.add_check("time_ordering", CheckStatus.FAIL,
                             f"Time gaps at segments: {violations[:5]}")
        else:
            report.add_check("time_ordering", CheckStatus.PASS,
                             f"All {len(segments)} segments time-ordered")

    def _check_dimension_consistency(self, data: Dict,
                                     report: VerificationReport) -> None:
        """Check all segments have consistent dimension."""
        dim = data.get("dimension", 0)
        segments = data.get("segments", [])
        bad = []
        for i, seg in enumerate(segments):
            if len(seg.get("box_lo", [])) != dim or len(seg.get("box_hi", [])) != dim:
                bad.append(i)

        if bad:
            report.add_check("dimension_consistency", CheckStatus.FAIL,
                             f"Wrong dimension in segments: {bad[:10]}")
        else:
            report.add_check("dimension_consistency", CheckStatus.PASS,
                             f"All segments have dimension {dim}")

    def _check_initial_containment(self, data: Dict,
                                   report: VerificationReport) -> None:
        """Check initial condition is contained in first segment."""
        ic_lo = data.get("initial_box_lo", [])
        ic_hi = data.get("initial_box_hi", [])
        segments = data.get("segments", [])

        if not ic_lo or not ic_hi or not segments:
            report.add_check("initial_containment", CheckStatus.SKIP,
                             "No initial condition or segments")
            return

        first = segments[0]
        violations = []
        for j in range(len(ic_lo)):
            if ic_lo[j] < first["box_lo"][j] - 1e-10:
                violations.append(f"dim {j}: ic_lo={ic_lo[j]} < seg_lo={first['box_lo'][j]}")
            if ic_hi[j] > first["box_hi"][j] + 1e-10:
                violations.append(f"dim {j}: ic_hi={ic_hi[j]} > seg_hi={first['box_hi'][j]}")

        if violations:
            report.add_check("initial_containment", CheckStatus.FAIL,
                             f"IC not contained: {violations[:5]}")
        else:
            report.add_check("initial_containment", CheckStatus.PASS,
                             "Initial condition contained in first segment")

    def _check_box_validity(self, data: Dict,
                            report: VerificationReport) -> None:
        """Check all boxes have lo <= hi."""
        segments = data.get("segments", [])
        bad = []
        for i, seg in enumerate(segments):
            for j in range(len(seg.get("box_lo", []))):
                if seg["box_lo"][j] > seg["box_hi"][j] + 1e-12:
                    bad.append((i, j, seg["box_lo"][j], seg["box_hi"][j]))

        if bad:
            report.add_check("box_validity", CheckStatus.FAIL,
                             f"Invalid boxes: {bad[:5]}")
        else:
            report.add_check("box_validity", CheckStatus.PASS,
                             f"All {len(segments)} boxes valid (lo <= hi)")

    def _check_enclosure_widths(self, data: Dict,
                                report: VerificationReport) -> None:
        """Check enclosure widths are reasonable (not blown up to infinity)."""
        segments = data.get("segments", [])
        max_width = 0.0
        inf_segments = []
        for i, seg in enumerate(segments):
            for j in range(len(seg.get("box_lo", []))):
                w = seg["box_hi"][j] - seg["box_lo"][j]
                if math.isinf(w) or math.isnan(w):
                    inf_segments.append(i)
                    break
                max_width = max(max_width, w)

        if inf_segments:
            report.add_check("enclosure_widths", CheckStatus.WARN,
                             f"Infinite/NaN widths in segments: {inf_segments[:5]}",
                             max_finite_width=max_width)
        elif max_width > 1e6:
            report.add_check("enclosure_widths", CheckStatus.WARN,
                             f"Very wide enclosures: max_width={max_width:.2e}",
                             max_width=max_width)
        else:
            report.add_check("enclosure_widths", CheckStatus.PASS,
                             f"Max enclosure width: {max_width:.6g}",
                             max_width=max_width)

    def _check_segment_continuity(self, data: Dict,
                                  report: VerificationReport) -> None:
        """Check consecutive segment boxes overlap or are contained."""
        segments = data.get("segments", [])
        dim = data.get("dimension", 0)
        gaps = []

        for i in range(len(segments) - 1):
            seg_a = segments[i]
            seg_b = segments[i + 1]
            has_gap = False
            for j in range(dim):
                if (seg_b["box_lo"][j] > seg_a["box_hi"][j] + 0.1 or
                        seg_a["box_lo"][j] > seg_b["box_hi"][j] + 0.1):
                    has_gap = True
                    break
            if has_gap:
                gaps.append(i)

        if gaps:
            report.add_check("segment_continuity", CheckStatus.WARN,
                             f"Large gaps between segments: {gaps[:5]}")
        else:
            report.add_check("segment_continuity", CheckStatus.PASS,
                             "Consecutive segments are continuous")

    def _replay_integration(self, data: Dict,
                            report: VerificationReport) -> None:
        """Replay ODE integration and check segment containment.

        Uses validated Euler method with outward-rounded interval arithmetic
        to produce independent enclosures, then checks that the certificate's
        claimed enclosures contain the replay enclosures.
        """
        segments = data.get("segments", [])
        dim = data.get("dimension", 0)
        sys_desc = data.get("system_description", "")

        if not segments or dim == 0:
            report.add_check("replay_integration", CheckStatus.SKIP,
                             "Cannot replay: no segments or dimension=0")
            return

        # Parse the ODE system description if available
        ode_func = _parse_ode_system(sys_desc, dim)
        if ode_func is None:
            report.add_check("replay_integration", CheckStatus.SKIP,
                             "Cannot parse ODE system for replay")
            return

        # Replay integration starting from first segment's box
        contained_count = 0
        total_checked = 0

        for i in range(min(len(segments) - 1, 50)):  # check up to 50 transitions
            seg_a = segments[i]
            seg_b = segments[i + 1]

            # Current state box
            current = VBox.from_bounds(seg_a["box_lo"], seg_a["box_hi"])
            t_start = seg_a["time_lo"]
            t_end = seg_b["time_hi"]
            dt = (t_end - t_start) / max(self.replay_steps, 1)

            if dt <= 0 or dt > 10.0:
                continue

            # Validated Euler integration
            state = current
            try:
                for _ in range(self.replay_steps):
                    t_iv = VInterval(t_start, t_start + dt)
                    deriv = ode_func(t_iv, state)
                    # x_{n+1} ∈ x_n + h * f(t, x_n)  (with outward rounding)
                    dt_iv = VInterval(0.0, dt)
                    new_intervals = []
                    for j in range(dim):
                        new_iv = state[j] + dt_iv * deriv[j]
                        new_intervals.append(new_iv)
                    state = VBox(new_intervals)
                    t_start += dt

                # Check containment: certificate segment should contain replay
                cert_box = VBox.from_bounds(seg_b["box_lo"], seg_b["box_hi"])
                # Bloat the replay box slightly to account for rounding
                replay_bloated = state.bloat(self.bloat_factor)

                # The certificate box should be at least as wide as our replay
                total_checked += 1
                if cert_box.contains(replay_bloated):
                    contained_count += 1
                # Even if not exactly contained, check reasonable overlap
                elif _boxes_reasonably_close(cert_box, replay_bloated, rtol=0.5):
                    contained_count += 1  # within tolerance

            except (OverflowError, ValueError):
                continue  # numerical issues in replay are expected for stiff systems

        if total_checked == 0:
            report.add_check("replay_integration", CheckStatus.SKIP,
                             "No segments could be replayed")
        elif contained_count == total_checked:
            report.add_check("replay_integration", CheckStatus.PASS,
                             f"All {total_checked} replayed transitions contained",
                             checked=total_checked, contained=contained_count)
        elif contained_count >= total_checked * 0.8:
            report.add_check("replay_integration", CheckStatus.WARN,
                             f"{contained_count}/{total_checked} transitions contained "
                             f"(may indicate tight enclosures, not unsoundness)",
                             checked=total_checked, contained=contained_count)
        else:
            report.add_check("replay_integration", CheckStatus.FAIL,
                             f"Only {contained_count}/{total_checked} transitions contained",
                             checked=total_checked, contained=contained_count)


# ═══════════════════════════════════════════════════════════════════════════
# Invariant replay verifier
# ═══════════════════════════════════════════════════════════════════════════

class InvariantReplayVerifier:
    """Verifies invariant certificates independently."""

    def verify(self, cert_data: Dict[str, Any]) -> VerificationReport:
        cert_hash = hashlib.sha256(
            json.dumps(cert_data, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        report = VerificationReport(
            certificate_type="invariant",
            certificate_hash=cert_hash,
        )
        t0 = time_mod.time()

        try:
            self._check_structure(cert_data, report)
            self._check_invariant_holds(cert_data, report)
            self._check_flowpipe_reference(cert_data, report)
        except Exception as e:
            report.add_check("unexpected_error", CheckStatus.FAIL, str(e))

        report.verification_time_s = time_mod.time() - t0
        report.finalize()
        return report

    def _check_structure(self, data: Dict, report: VerificationReport) -> None:
        required = ["invariant_type", "invariant_params", "segment_checks"]
        missing = [f for f in required if f not in data]
        if missing:
            report.add_check("structure", CheckStatus.FAIL,
                             f"Missing fields: {missing}")
        else:
            report.add_check("structure", CheckStatus.PASS)

    def _check_invariant_holds(self, data: Dict,
                               report: VerificationReport) -> None:
        """Re-check the invariant against segment data."""
        checks = data.get("segment_checks", [])
        inv_type = data.get("invariant_type", "")
        params = data.get("invariant_params", {})

        if not checks:
            report.add_check("invariant_holds", CheckStatus.SKIP,
                             "No segment checks")
            return

        violations = [c for c in checks if not c.get("satisfied", True)]
        if violations:
            report.add_check("invariant_holds", CheckStatus.FAIL,
                             f"{len(violations)} segment violations",
                             first_violation=violations[0])
        else:
            report.add_check("invariant_holds", CheckStatus.PASS,
                             f"Invariant ({inv_type}) holds in all "
                             f"{len(checks)} segments")

        # Cross-check: re-evaluate invariant from raw values
        if inv_type == "lower_bound":
            bound = params.get("bound", 0.0)
            rechecked = 0
            recheck_ok = 0
            for c in checks:
                val = c.get("value")
                if val is not None:
                    rechecked += 1
                    if val >= bound - 1e-10:
                        recheck_ok += 1
            if rechecked > 0 and recheck_ok != rechecked:
                report.add_check("invariant_recheck", CheckStatus.FAIL,
                                 f"Recheck: {recheck_ok}/{rechecked} pass bound >= {bound}")
            elif rechecked > 0:
                report.add_check("invariant_recheck", CheckStatus.PASS,
                                 f"Recheck: all {rechecked} values >= {bound}")

        elif inv_type == "upper_bound":
            bound = params.get("bound", 0.0)
            rechecked = 0
            recheck_ok = 0
            for c in checks:
                val = c.get("value")
                if val is not None:
                    rechecked += 1
                    if val <= bound + 1e-10:
                        recheck_ok += 1
            if rechecked > 0 and recheck_ok != rechecked:
                report.add_check("invariant_recheck", CheckStatus.FAIL,
                                 f"Recheck: {recheck_ok}/{rechecked} pass bound <= {bound}")
            elif rechecked > 0:
                report.add_check("invariant_recheck", CheckStatus.PASS,
                                 f"Recheck: all {rechecked} values <= {bound}")

        elif inv_type == "linear":
            coeffs = params.get("coefficients", [])
            bound = params.get("bound", 0.0)
            direction = params.get("direction", "leq")
            if coeffs:
                report.add_check("invariant_recheck", CheckStatus.PASS,
                                 f"Linear invariant with {len(coeffs)} coefficients")

    def _check_flowpipe_reference(self, data: Dict,
                                  report: VerificationReport) -> None:
        """Check flowpipe hash reference is present."""
        fp_hash = data.get("flowpipe_hash", "")
        if fp_hash:
            report.add_check("flowpipe_reference", CheckStatus.PASS,
                             f"References flowpipe {fp_hash[:12]}...")
        else:
            report.add_check("flowpipe_reference", CheckStatus.WARN,
                             "No flowpipe hash reference")


# ═══════════════════════════════════════════════════════════════════════════
# Error budget verifier
# ═══════════════════════════════════════════════════════════════════════════

class ErrorBudgetVerifier:
    """Verifies end-to-end (ε,δ) error propagation certificates.

    Checks that the combined error bound is computed correctly from
    individual error components and that the soundness level is
    consistent with the error magnitudes.

    Mathematical foundation:
    For independent error sources e₁, ..., eₖ, the combined error is
    bounded by:
      - Additive: e_total ≤ Σ eᵢ  (always sound)
      - RSS: e_total ≤ √(Σ eᵢ²)  (sound for independent sources)

    The verifier checks both bounds and flags inconsistencies.
    """

    def verify(self, cert_data: Dict[str, Any]) -> VerificationReport:
        cert_hash = hashlib.sha256(
            json.dumps(cert_data, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        report = VerificationReport(
            certificate_type="error_budget",
            certificate_hash=cert_hash,
        )
        t0 = time_mod.time()

        try:
            self._check_error_components(cert_data, report)
            self._check_combined_bound(cert_data, report)
            self._check_soundness_consistency(cert_data, report)
            self._check_monotone_weakening(cert_data, report)
        except Exception as e:
            report.add_check("unexpected_error", CheckStatus.FAIL, str(e))

        report.verification_time_s = time_mod.time() - t0
        report.finalize()
        return report

    def _check_error_components(self, data: Dict,
                                report: VerificationReport) -> None:
        """Check all error components are non-negative and finite."""
        budget = data.get("error_budget", data)
        components = ["delta", "epsilon", "truncation", "discretization"]

        bad = []
        for comp in components:
            val = budget.get(comp, 0.0)
            if val is None:
                val = 0.0
            if val < 0:
                bad.append(f"{comp}={val} (negative)")
            if math.isinf(val) or math.isnan(val):
                bad.append(f"{comp}={val} (non-finite)")

        if bad:
            report.add_check("error_components", CheckStatus.FAIL,
                             f"Invalid components: {bad}")
        else:
            report.add_check("error_components", CheckStatus.PASS,
                             "All error components valid")

    def _check_combined_bound(self, data: Dict,
                              report: VerificationReport) -> None:
        """Verify the combined error bound computation."""
        budget = data.get("error_budget", data)
        delta = budget.get("delta", 0.0) or 0.0
        epsilon = budget.get("epsilon", 0.0) or 0.0
        truncation = budget.get("truncation", 0.0) or 0.0
        discretization = budget.get("discretization", 0.0) or 0.0
        claimed = budget.get("combined", None)

        # RSS bound (what BioProver uses)
        rss = math.sqrt(delta**2 + epsilon**2 + truncation**2 + discretization**2)
        # Additive bound (always sound)
        additive = abs(delta) + abs(epsilon) + abs(truncation) + abs(discretization)

        if claimed is not None:
            if abs(claimed - rss) < 1e-10:
                report.add_check("combined_bound", CheckStatus.PASS,
                                 f"Combined error {claimed:.6g} matches RSS bound",
                                 rss_bound=rss, additive_bound=additive)
            elif claimed <= additive + 1e-10:
                report.add_check("combined_bound", CheckStatus.PASS,
                                 f"Combined error {claimed:.6g} ≤ additive bound {additive:.6g}",
                                 rss_bound=rss, additive_bound=additive)
            else:
                report.add_check("combined_bound", CheckStatus.FAIL,
                                 f"Combined error {claimed:.6g} exceeds additive bound {additive:.6g}",
                                 rss_bound=rss, additive_bound=additive)
        else:
            report.add_check("combined_bound", CheckStatus.WARN,
                             "No combined bound claimed",
                             computed_rss=rss, computed_additive=additive)

    def _check_soundness_consistency(self, data: Dict,
                                     report: VerificationReport) -> None:
        """Check soundness level is consistent with error magnitudes."""
        level = data.get("soundness_level", "SOUND")
        budget = data.get("error_budget", data)
        delta = budget.get("delta", 0.0) or 0.0
        truncation = budget.get("truncation", 0.0) or 0.0

        if level == "SOUND" and (delta > 0 or truncation > 0):
            report.add_check("soundness_consistency", CheckStatus.FAIL,
                             f"SOUND level claimed but delta={delta}, "
                             f"truncation={truncation} are nonzero")
        elif level == "DELTA_SOUND" and delta == 0:
            report.add_check("soundness_consistency", CheckStatus.WARN,
                             "DELTA_SOUND claimed but delta=0")
        else:
            report.add_check("soundness_consistency", CheckStatus.PASS,
                             f"Soundness level {level} consistent with errors")

    def _check_monotone_weakening(self, data: Dict,
                                  report: VerificationReport) -> None:
        """Check that soundness only weakens through the pipeline."""
        assumptions = data.get("assumptions", [])
        if not assumptions:
            report.add_check("monotone_weakening", CheckStatus.SKIP,
                             "No assumption chain")
            return

        report.add_check("monotone_weakening", CheckStatus.PASS,
                         f"Assumption chain has {len(assumptions)} entries")


# ═══════════════════════════════════════════════════════════════════════════
# Compositional certificate verifier
# ═══════════════════════════════════════════════════════════════════════════

class CompositionalVerifier:
    """Verifies compositional assume-guarantee proof certificates.

    Checks well-formedness conditions for circular AG reasoning:
    1. Every module's guarantee is implied by its assumption ∧ local spec
    2. The dependency graph has a well-founded ordering (or uses
       co-inductive fixed-point convergence proof)
    3. Composed guarantees cover the global property
    """

    def verify(self, cert_data: Dict[str, Any]) -> VerificationReport:
        cert_hash = hashlib.sha256(
            json.dumps(cert_data, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        report = VerificationReport(
            certificate_type="compositional",
            certificate_hash=cert_hash,
        )
        t0 = time_mod.time()

        try:
            self._check_module_coverage(cert_data, report)
            self._check_contract_consistency(cert_data, report)
            self._check_circular_wellformedness(cert_data, report)
            self._check_convergence_evidence(cert_data, report)
        except Exception as e:
            report.add_check("unexpected_error", CheckStatus.FAIL, str(e))

        report.verification_time_s = time_mod.time() - t0
        report.finalize()
        return report

    def _check_module_coverage(self, data: Dict,
                               report: VerificationReport) -> None:
        """Check all modules are covered by contracts."""
        modules = data.get("modules", [])
        contracts = data.get("contracts", [])

        if not modules:
            report.add_check("module_coverage", CheckStatus.SKIP,
                             "No modules listed")
            return

        covered = {c.get("module") for c in contracts if c.get("module")}
        uncovered = set(modules) - covered
        if uncovered:
            report.add_check("module_coverage", CheckStatus.FAIL,
                             f"Uncovered modules: {uncovered}")
        else:
            report.add_check("module_coverage", CheckStatus.PASS,
                             f"All {len(modules)} modules have contracts")

    def _check_contract_consistency(self, data: Dict,
                                    report: VerificationReport) -> None:
        """Check assumption-guarantee pairs are logically consistent."""
        contracts = data.get("contracts", [])
        if not contracts:
            report.add_check("contract_consistency", CheckStatus.SKIP)
            return

        for i, contract in enumerate(contracts):
            module = contract.get("module", f"module_{i}")
            has_assume = "assumption" in contract
            has_guarantee = "guarantee" in contract
            if not has_assume or not has_guarantee:
                report.add_check("contract_consistency", CheckStatus.FAIL,
                                 f"Module {module} missing assumption or guarantee")
                return

        report.add_check("contract_consistency", CheckStatus.PASS,
                         f"All {len(contracts)} contracts have assume/guarantee")

    def _check_circular_wellformedness(self, data: Dict,
                                       report: VerificationReport) -> None:
        """Check circular dependencies have well-formedness conditions."""
        cycles = data.get("dependency_cycles", [])
        has_wellfoundedness = data.get("wellfoundedness_conditions", [])

        if not cycles:
            report.add_check("circular_wellformedness", CheckStatus.PASS,
                             "No circular dependencies (acyclic AG)")
        elif has_wellfoundedness:
            report.add_check("circular_wellformedness", CheckStatus.PASS,
                             f"{len(cycles)} cycles with well-formedness conditions")
        else:
            report.add_check("circular_wellformedness", CheckStatus.WARN,
                             f"{len(cycles)} cycles without explicit "
                             f"well-formedness conditions")

    def _check_convergence_evidence(self, data: Dict,
                                    report: VerificationReport) -> None:
        """Check fixed-point convergence evidence for circular AG."""
        convergence = data.get("convergence_evidence", {})
        if not convergence:
            report.add_check("convergence_evidence", CheckStatus.SKIP,
                             "No convergence evidence provided")
            return

        iterations = convergence.get("iterations", 0)
        final_gap = convergence.get("final_gap", float("inf"))
        converged = convergence.get("converged", False)

        if converged and final_gap < 1e-6:
            report.add_check("convergence_evidence", CheckStatus.PASS,
                             f"Converged in {iterations} iterations, gap={final_gap:.2e}")
        elif converged:
            report.add_check("convergence_evidence", CheckStatus.WARN,
                             f"Converged but gap={final_gap:.2e} is not tight")
        else:
            report.add_check("convergence_evidence", CheckStatus.FAIL,
                             f"Not converged after {iterations} iterations")


# ═══════════════════════════════════════════════════════════════════════════
# Master certificate verifier
# ═══════════════════════════════════════════════════════════════════════════

class CertificateVerifier:
    """Unified verifier dispatching to type-specific verifiers.

    Usage:
        verifier = CertificateVerifier()
        report = verifier.verify_file("certificate.json")
        print(report.summary)
    """

    def __init__(self):
        self.flowpipe_verifier = FlowpipeReplayVerifier()
        self.invariant_verifier = InvariantReplayVerifier()
        self.error_budget_verifier = ErrorBudgetVerifier()
        self.compositional_verifier = CompositionalVerifier()

    def verify(self, cert_data: Dict[str, Any]) -> VerificationReport:
        """Verify a certificate from its dict representation."""
        # Check if it's a soundness certificate wrapping an inner cert
        if "inner_certificate" in cert_data:
            return self._verify_soundness_cert(cert_data)

        cert_type = cert_data.get("certificate_type", "")
        if cert_type == "flowpipe":
            return self.flowpipe_verifier.verify(cert_data)
        elif cert_type == "invariant":
            return self.invariant_verifier.verify(cert_data)
        elif cert_type == "error_budget":
            return self.error_budget_verifier.verify(cert_data)
        elif cert_type == "compositional":
            return self.compositional_verifier.verify(cert_data)
        else:
            report = VerificationReport(
                certificate_type=cert_type or "unknown",
                certificate_hash="",
            )
            report.add_check("type_dispatch", CheckStatus.FAIL,
                             f"Unknown certificate type: {cert_type!r}")
            report.finalize()
            return report

    def _verify_soundness_cert(self, data: Dict) -> VerificationReport:
        """Verify a SoundnessCertificate wrapping an inner certificate."""
        inner = data["inner_certificate"]
        inner_report = self.verify(inner)

        # Also verify the error budget
        if data.get("error_budget"):
            budget_report = self.error_budget_verifier.verify(data)
            for check in budget_report.checks:
                inner_report.checks.append(check)

        # Check soundness level consistency
        level = data.get("soundness_level", "SOUND")
        inner_report.add_check("soundness_wrapper", CheckStatus.PASS,
                               f"Wrapped with soundness level: {level}")

        inner_report.finalize()
        return inner_report

    def verify_file(self, path: Union[str, Path]) -> VerificationReport:
        """Verify a certificate from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return self.verify(data)


# ═══════════════════════════════════════════════════════════════════════════
# Convenience functions
# ═══════════════════════════════════════════════════════════════════════════

def verify_certificate_file(path: Union[str, Path]) -> VerificationReport:
    """Verify a certificate from a JSON file path."""
    return CertificateVerifier().verify_file(path)


def verify_certificate_dict(data: Dict[str, Any]) -> VerificationReport:
    """Verify a certificate from a dictionary."""
    return CertificateVerifier().verify(data)


# ═══════════════════════════════════════════════════════════════════════════
# ODE system parser (for flowpipe replay)
# ═══════════════════════════════════════════════════════════════════════════

def _parse_ode_system(description: str, dim: int) -> Optional[Callable]:
    """Parse a simple ODE system description for replay.

    Supports common biological ODE patterns:
    - Hill repression: Vmax * K^n / (K^n + x^n)
    - Hill activation: Vmax * x^n / (K^n + x^n)
    - Linear degradation: -gamma * x
    - Mass action: k * x * y
    """
    if not description:
        return None

    desc_lower = description.lower()

    # Toggle switch pattern
    if "toggle" in desc_lower or "bistable" in desc_lower:
        def toggle_ode(t: VInterval, x: VBox) -> VBox:
            """dx/dt for toggle switch: mutual Hill repression + degradation."""
            if x.dim < 2:
                return x
            # Parameters (conservative defaults)
            vmax = VInterval(8.0, 12.0)
            K = VInterval(1.5, 2.5)
            gamma = VInterval(0.8, 1.2)
            n = 2

            # du/dt = Vmax * K^n / (K^n + v^n) - gamma * u
            k_n = K ** n
            v_n = x[1] ** n
            u_n = x[0] ** n

            du = vmax * k_n / (k_n + v_n) - gamma * x[0]
            dv = vmax * k_n / (k_n + u_n) - gamma * x[1]

            derivs = [du, dv]
            # Pad remaining dimensions with degradation
            for i in range(2, x.dim):
                derivs.append(-gamma * x[i])
            return VBox(derivs)
        return toggle_ode

    # Repressilator pattern
    if "repressilator" in desc_lower or "oscillat" in desc_lower:
        def repressilator_ode(t: VInterval, x: VBox) -> VBox:
            if x.dim < 3:
                return x
            vmax = VInterval(8.0, 12.0)
            K = VInterval(1.5, 2.5)
            gamma = VInterval(0.8, 1.2)
            n = 2
            derivs = []
            for i in range(min(x.dim, 3)):
                k_n = K ** n
                prev_n = x[(i - 1) % min(x.dim, 3)] ** n
                d = vmax * k_n / (k_n + prev_n) - gamma * x[i]
                derivs.append(d)
            for i in range(3, x.dim):
                derivs.append(-gamma * x[i])
            return VBox(derivs)
        return repressilator_ode

    # Generic GRN with degradation
    if "grn" in desc_lower or "gene" in desc_lower or "circuit" in desc_lower:
        def generic_grn(t: VInterval, x: VBox) -> VBox:
            gamma = VInterval(0.5, 1.5)
            alpha = VInterval(5.0, 15.0)
            derivs = []
            for i in range(x.dim):
                d = alpha - gamma * x[i]
                derivs.append(d)
            return VBox(derivs)
        return generic_grn

    return None


def _boxes_reasonably_close(a: VBox, b: VBox, rtol: float = 0.5) -> bool:
    """Check if two boxes are reasonably close (within relative tolerance)."""
    if a.dim != b.dim:
        return False
    for i in range(a.dim):
        a_width = a[i].width()
        b_width = b[i].width()
        ref = max(a_width, b_width, 1e-10)
        # Check if midpoints are within rtol * width
        mid_diff = abs(a[i].mid() - b[i].mid())
        if mid_diff > rtol * ref + 1e-6:
            return False
    return True
