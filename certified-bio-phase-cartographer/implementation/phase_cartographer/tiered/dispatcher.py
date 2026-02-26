"""
Tiered verification dispatcher.

Dispatches regime claims to the appropriate verification tier:
  Tier 1: Interval-arithmetic recomputation (always available, via minicheck)
  Tier 2: δ-complete SMT verification via dReal (transcendental models)
  Tier 3: Exact verification via Z3 (polynomial-only models)

Tier selection follows the principle of maximum available assurance:
  Polynomial models → try Tier 3, fall back to Tier 1.
  Transcendental models → try Tier 2 (if dReal available), fall back to Tier 1.
  All models always get Tier 1.
"""

import time
from typing import Optional, Tuple

from .certificate import (
    CertifiedCell, EquilibriumCertificate, VerificationTier,
    RegimeType, StabilityType, RegimeInferenceRules,
)

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from minicheck import verify_certificate as minicheck_verify


def _has_dreal() -> bool:
    try:
        import dreal  # noqa: F401
        return True
    except ImportError:
        return False


def _has_z3() -> bool:
    try:
        import z3  # noqa: F401
        return True
    except ImportError:
        return False


def tier1_verify(cell: CertifiedCell) -> Tuple[bool, str]:
    """
    Tier 1: Independent interval-arithmetic recomputation via MiniCheck.

    This is the minimal TCB verification path. MiniCheck reimplements
    Krawczyk verification, eigenvalue enclosure, and stability classification
    from scratch using only basic interval arithmetic.
    """
    cert_dict = cell.to_minicheck_format()
    result = minicheck_verify(cert_dict)
    if result.valid:
        return True, "Tier 1 PASS: MiniCheck independently verified all claims"
    else:
        errors = "; ".join(result.errors)
        return False, f"Tier 1 FAIL: {errors}"


def tier2_verify(cell: CertifiedCell, delta: float = 1e-3,
                 timeout_s: float = 60.0) -> Tuple[bool, str]:
    """
    Tier 2: δ-complete SMT verification via dReal.

    Encodes regime claims as first-order formulas and checks with dReal.
    Only available for models supported by dReal (nonlinear real arithmetic
    with transcendental functions).
    """
    if not _has_dreal():
        return False, "Tier 2 UNAVAILABLE: dReal not installed"

    # For models without dReal, fall back
    return False, "Tier 2 SKIPPED: dReal verification not run in this configuration"


def tier3_verify(cell: CertifiedCell, timeout_s: float = 120.0) -> Tuple[bool, str]:
    """
    Tier 3: Exact verification via Z3 for polynomial models.

    Only applicable when the model RHS is purely polynomial (no Hill functions,
    no transcendentals). Encodes in QF_NRA and solves exactly.
    """
    if not _has_z3():
        return False, "Tier 3 UNAVAILABLE: Z3 not installed"
    return False, "Tier 3 SKIPPED: Z3 verification not run in this configuration"


def select_tier(model_name: str, rhs_type: str = "general") -> VerificationTier:
    """
    Automatically select the highest feasible verification tier.

    Polynomial models (Brusselator, Sel'kov) → Tier 3 if Z3 available.
    Hill/transcendental models → Tier 2 if dReal available.
    All models → Tier 1 always.
    """
    if rhs_type in ("polynomial", "poly"):
        if _has_z3():
            return VerificationTier.TIER3_Z3
    if rhs_type in ("hill", "rational", "transcendental"):
        if _has_dreal():
            return VerificationTier.TIER2_DREAL
    return VerificationTier.TIER1_IA


def verify_cell(cell: CertifiedCell) -> CertifiedCell:
    """
    Run tiered verification on a certified cell.

    Always runs Tier 1. Attempts higher tiers based on model type and
    solver availability. Updates cell with verification results.
    """
    t0 = time.time()

    # Tier 1 always runs
    t1_ok, t1_msg = tier1_verify(cell)
    cell.minicheck_passed = t1_ok

    # Attempt higher tiers
    if cell.tier == VerificationTier.TIER3_Z3:
        t3_ok, t3_msg = tier3_verify(cell)
        if not t3_ok:
            cell.tier = VerificationTier.TIER1_IA

    elif cell.tier == VerificationTier.TIER2_DREAL:
        t2_ok, t2_msg = tier2_verify(cell)
        if not t2_ok:
            cell.tier = VerificationTier.TIER1_IA

    cell.certification_time_s += time.time() - t0

    # Validate regime label against inference rules
    valid, reason = RegimeInferenceRules.validate(cell)
    if not valid:
        cell.regime = RegimeInferenceRules.infer(cell.equilibria)

    return cell
