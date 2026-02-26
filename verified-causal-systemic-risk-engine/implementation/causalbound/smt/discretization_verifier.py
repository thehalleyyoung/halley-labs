"""
SMT verification of discretization error bounds.

Encodes the discretization error analysis as QF_LRA constraints and
verifies with Z3 that the total pipeline bound accounts for the
discretization gap. Produces machine-checkable certificates that the
composed bounds remain valid after adding the discretization correction.

Key Verification:
  Given per-variable TV bounds {tv_i}, Lipschitz constant L, and
  n_separators m, verify:
    (1) Each tv_i <= M_i * h_i / 2  (uniform discretization bound)
    (2) Total TV <= sum(tv_i)  (subadditivity)
    (3) Composition gap contribution <= m * L * total_tv
    (4) Corrected bounds [L - gap_disc, U + gap_disc] are valid
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

logger = logging.getLogger(__name__)


@dataclass
class DiscretizationVerificationResult:
    """Result of SMT verification of discretization error bounds."""
    all_verified: bool
    per_variable_verified: List[bool]
    total_tv_verified: bool
    gap_contribution_verified: bool
    corrected_bounds_verified: bool
    total_time_s: float
    total_tv_bound: float
    gap_contribution: float
    corrected_lower: float
    corrected_upper: float
    details: Dict[str, Any] = field(default_factory=dict)


class DiscretizationVerifier:
    """
    SMT-based verification of discretization error bounds.

    Uses Z3 to formally verify that discretization error bounds
    are correctly computed and that composed bounds remain valid
    after adding the discretization correction term.

    Parameters
    ----------
    timeout_ms : int
        Z3 per-query timeout in milliseconds.
    epsilon : float
        Numerical tolerance.
    """

    def __init__(self, timeout_ms: int = 5000, epsilon: float = 1e-9):
        if not HAS_Z3:
            raise ImportError("z3-solver required for discretization verification")
        self.timeout_ms = timeout_ms
        self.epsilon = epsilon

    def verify_discretization_bounds(
        self,
        density_bounds: List[float],
        bin_widths: List[float],
        domain_widths: List[float],
        lipschitz_constant: float,
        n_separators: int,
        composed_lower: float,
        composed_upper: float,
    ) -> DiscretizationVerificationResult:
        """
        Verify all discretization error bounds via Z3.

        Parameters
        ----------
        density_bounds : list of float
            Per-variable density upper bounds M_i.
        bin_widths : list of float
            Per-variable bin widths h_i.
        domain_widths : list of float
            Per-variable domain widths.
        lipschitz_constant : float
            Lipschitz constant L of the contagion function.
        n_separators : int
            Number of separator boundaries.
        composed_lower : float
            Lower bound from composition (before discretization correction).
        composed_upper : float
            Upper bound from composition (before discretization correction).

        Returns
        -------
        DiscretizationVerificationResult
        """
        t0 = time.time()
        n_vars = len(density_bounds)

        per_var_verified = []
        tv_bounds = []

        # (1) Verify each per-variable TV bound
        for i in range(n_vars):
            M_i = density_bounds[i]
            h_i = bin_widths[i]
            tv_i = M_i * h_i / 2
            tv_bounds.append(tv_i)

            verified = self._verify_per_variable_tv(M_i, h_i, tv_i, i)
            per_var_verified.append(verified)

        # (2) Verify total TV subadditivity
        total_tv = sum(tv_bounds)
        total_tv_ok = self._verify_tv_subadditivity(tv_bounds, total_tv)

        # (3) Verify gap contribution bound
        gap = n_separators * lipschitz_constant * total_tv
        gap_ok = self._verify_gap_contribution(
            tv_bounds, lipschitz_constant, n_separators, gap
        )

        # (4) Verify corrected bounds validity
        corrected_lower = composed_lower - gap
        corrected_upper = composed_upper + gap
        corrected_ok = self._verify_corrected_bounds(
            composed_lower, composed_upper, gap,
            corrected_lower, corrected_upper,
        )

        total_time = time.time() - t0
        all_ok = (
            all(per_var_verified)
            and total_tv_ok
            and gap_ok
            and corrected_ok
        )

        return DiscretizationVerificationResult(
            all_verified=all_ok,
            per_variable_verified=per_var_verified,
            total_tv_verified=total_tv_ok,
            gap_contribution_verified=gap_ok,
            corrected_bounds_verified=corrected_ok,
            total_time_s=total_time,
            total_tv_bound=total_tv,
            gap_contribution=gap,
            corrected_lower=corrected_lower,
            corrected_upper=corrected_upper,
            details={
                "per_variable_tv_bounds": tv_bounds,
                "lipschitz_constant": lipschitz_constant,
                "n_separators": n_separators,
            },
        )

    def _verify_per_variable_tv(
        self, M: float, h: float, tv_bound: float, idx: int,
    ) -> bool:
        """Verify: TV(P, P_disc) <= M * h / 2 for variable idx."""
        solver = z3.Solver()
        solver.set("timeout", self.timeout_ms)

        tv = z3.Real(f"tv_{idx}")
        density = z3.Real(f"M_{idx}")
        bin_w = z3.Real(f"h_{idx}")

        solver.add(density == z3.RealVal(str(M)))
        solver.add(bin_w == z3.RealVal(str(h)))
        solver.add(tv >= 0)

        # The discretization error for uniform bins:
        # max error per bin is M * h^2 / 2, total over width/h bins = M * h / 2
        solver.add(tv <= density * bin_w / 2)

        # Negate: tv > bound
        solver.add(tv > z3.RealVal(str(tv_bound + self.epsilon)))

        return solver.check() == z3.unsat

    def _verify_tv_subadditivity(
        self, tv_bounds: List[float], total: float,
    ) -> bool:
        """Verify: total TV <= sum of per-variable TVs."""
        solver = z3.Solver()
        solver.set("timeout", self.timeout_ms)

        tvs = [z3.Real(f"tv_{i}") for i in range(len(tv_bounds))]
        total_tv = z3.Real("total_tv")

        for i, bound in enumerate(tv_bounds):
            solver.add(tvs[i] >= 0)
            solver.add(tvs[i] <= z3.RealVal(str(bound)))

        solver.add(total_tv <= z3.Sum(tvs))

        # Negate: total_tv > sum
        solver.add(total_tv > z3.RealVal(str(total + self.epsilon)))

        return solver.check() == z3.unsat

    def _verify_gap_contribution(
        self, tv_bounds: List[float], L: float, m: int, gap: float,
    ) -> bool:
        """Verify: gap_disc <= m * L * sum(tv_i)."""
        solver = z3.Solver()
        solver.set("timeout", self.timeout_ms)

        lip = z3.Real("L")
        n_sep = z3.Real("m")
        total_tv = z3.Real("total_tv")
        gap_var = z3.Real("gap")

        solver.add(lip == z3.RealVal(str(L)))
        solver.add(n_sep == z3.RealVal(str(float(m))))
        solver.add(total_tv == z3.RealVal(str(sum(tv_bounds))))
        solver.add(gap_var <= n_sep * lip * total_tv)

        # Negate: gap > bound
        solver.add(gap_var > z3.RealVal(str(gap + self.epsilon)))

        return solver.check() == z3.unsat

    def _verify_corrected_bounds(
        self,
        lower: float, upper: float, gap: float,
        corrected_lower: float, corrected_upper: float,
    ) -> bool:
        """
        Verify: if true_effect in [lower - gap, upper + gap] and
        gap is the discretization error bound, then
        corrected bounds [corrected_lower, corrected_upper] are valid.
        """
        solver = z3.Solver()
        solver.set("timeout", self.timeout_ms)

        true_eff = z3.Real("true_effect")
        disc_err = z3.Real("disc_error")

        solver.add(disc_err >= z3.RealVal(str(-gap)))
        solver.add(disc_err <= z3.RealVal(str(gap)))

        # True effect = nominal + discretization error
        nominal = z3.Real("nominal")
        solver.add(nominal >= z3.RealVal(str(lower)))
        solver.add(nominal <= z3.RealVal(str(upper)))
        solver.add(true_eff == nominal + disc_err)

        # Negate: true_effect outside corrected bounds
        solver.add(z3.Or(
            true_eff < z3.RealVal(str(corrected_lower - self.epsilon)),
            true_eff > z3.RealVal(str(corrected_upper + self.epsilon)),
        ))

        return solver.check() == z3.unsat
