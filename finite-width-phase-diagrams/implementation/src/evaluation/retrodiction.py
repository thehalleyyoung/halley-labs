"""Retrodiction validation against known theoretical results.

Validates pipeline predictions against established analytical results
from the neural network theory literature: Chizat & Bach lazy/rich
boundary, Saxe linear network dynamics, µP scaling exponents, and
mean-field kernel fixed points.

Provides:
  - KnownResult: specification of an expected theoretical value
  - RetrodictionResult: comparison of computed vs expected value
  - RetrodictionValidator: runs all retrodiction checks
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import linalg as sp_linalg


# ======================================================================
# Data classes
# ======================================================================


@dataclass
class KnownResult:
    """A known theoretical result for retrodiction validation.

    Parameters
    ----------
    name : str
        Short identifier for this result.
    description : str
        Human-readable description of the expected behavior.
    expected_value : float
        Numerically expected value.
    tolerance : float
        Acceptable relative deviation.
    reference : str
        Literature reference (author, year, theorem/equation).
    parameter_regime : dict
        Parameter values at which the result holds.
    """

    name: str = ""
    description: str = ""
    expected_value: float = 0.0
    tolerance: float = 0.1
    reference: str = ""
    parameter_regime: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrodictionResult:
    """Result of validating one known result.

    Parameters
    ----------
    known : KnownResult
        The expected result being checked.
    computed_value : float
        Value produced by the pipeline.
    deviation : float
        Relative deviation from expected: |computed - expected| / |expected|.
    within_tolerance : bool
        Whether deviation is within tolerance.
    details : dict
        Additional diagnostic information.
    """

    known: KnownResult = field(default_factory=KnownResult)
    computed_value: float = 0.0
    deviation: float = 0.0
    within_tolerance: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


# ======================================================================
# Retrodiction validator
# ======================================================================


class RetrodictionValidator:
    """Validate pipeline predictions against known theoretical results.

    Contains a built-in database of known results from the neural
    network theory literature and methods to check each one.

    Examples
    --------
    >>> validator = RetrodictionValidator()
    >>> results = validator.run_all(compute_fns)
    >>> print(validator.summary_report(results))
    """

    def __init__(self) -> None:
        self._known = self._known_results()

    # ------------------------------------------------------------------
    # Known results database
    # ------------------------------------------------------------------

    @staticmethod
    def _known_results() -> List[KnownResult]:
        """Built-in database of known theoretical results.

        Returns
        -------
        results : list of KnownResult
        """
        return [
            KnownResult(
                name="chizat_bach_boundary",
                description=(
                    "Chizat & Bach (2019): the lazy-to-rich transition "
                    "occurs when η·n ~ O(1), where η is the learning "
                    "rate and n is the width.  At the boundary, the "
                    "product η*n should be of order 1."
                ),
                expected_value=1.0,
                tolerance=0.5,
                reference="Chizat & Bach, 2019, Theorem 3.1",
                parameter_regime={
                    "scaling": "ntk",
                    "boundary_type": "lazy_rich",
                    "observable": "eta_times_n",
                },
            ),
            KnownResult(
                name="saxe_linear_dynamics",
                description=(
                    "Saxe et al. (2014): linear networks with "
                    "orthogonal initialization converge exponentially "
                    "with rate proportional to the singular value. "
                    "The convergence exponent should equal 2·σ for "
                    "the dominant mode with singular value σ."
                ),
                expected_value=2.0,
                tolerance=0.2,
                reference="Saxe et al., 2014, Eq. 12",
                parameter_regime={
                    "activation": "linear",
                    "depth": 2,
                    "init": "orthogonal",
                    "observable": "convergence_rate_ratio",
                },
            ),
            KnownResult(
                name="mup_scaling_exponents",
                description=(
                    "Yang & Hu (2021) µP: under maximal update "
                    "parametrization, the LR scales as 1/n and "
                    "init scale as 1/√n.  The exponent for LR "
                    "scaling should be -1."
                ),
                expected_value=-1.0,
                tolerance=0.15,
                reference="Yang & Hu, 2021, Table 3",
                parameter_regime={
                    "parametrization": "mup",
                    "observable": "lr_scaling_exponent",
                },
            ),
            KnownResult(
                name="kernel_fixed_point",
                description=(
                    "Mean-field kernel fixed point: for ReLU "
                    "activations at infinite depth, the diagonal "
                    "of the kernel converges to a fixed point "
                    "q* ≈ 1/(2π) · (π - 1) under standard "
                    "parametrization with σ_w² = 2, σ_b² = 0."
                ),
                expected_value=1.0,
                tolerance=0.1,
                reference=(
                    "Poole et al., 2016; Schoenholz et al., 2017, Eq. 5"
                ),
                parameter_regime={
                    "activation": "relu",
                    "sigma_w_sq": 2.0,
                    "sigma_b_sq": 0.0,
                    "depth": "infinite",
                    "observable": "kernel_fixed_point_q_star",
                },
            ),
        ]

    # ------------------------------------------------------------------
    # Individual validations
    # ------------------------------------------------------------------

    def validate_chizat_bach(
        self,
        compute_fn: Callable[..., float],
    ) -> RetrodictionResult:
        """Reproduce the Chizat & Bach lazy-to-rich boundary scaling.

        The compute function should accept keyword arguments matching
        the parameter regime and return the product η*n at the
        empirically determined boundary.

        Parameters
        ----------
        compute_fn : callable
            Function(**parameter_regime) -> float.

        Returns
        -------
        result : RetrodictionResult
        """
        known = self._known[0]
        try:
            computed = float(compute_fn(**known.parameter_regime))
        except Exception as exc:
            return RetrodictionResult(
                known=known,
                computed_value=float("nan"),
                deviation=float("inf"),
                within_tolerance=False,
                details={"error": str(exc)},
            )

        deviation = self._relative_deviation(computed, known.expected_value)
        within = deviation <= known.tolerance

        return RetrodictionResult(
            known=known,
            computed_value=computed,
            deviation=deviation,
            within_tolerance=within,
            details={
                "expected_product": known.expected_value,
                "computed_product": computed,
            },
        )

    def validate_saxe_dynamics(
        self,
        compute_fn: Callable[..., float],
    ) -> RetrodictionResult:
        """Reproduce Saxe et al. linear network convergence rate.

        The compute function should return the ratio of convergence
        rate to singular value, which should be ~2 for the dominant mode.

        Parameters
        ----------
        compute_fn : callable
            Function(**parameter_regime) -> float.

        Returns
        -------
        result : RetrodictionResult
        """
        known = self._known[1]
        try:
            computed = float(compute_fn(**known.parameter_regime))
        except Exception as exc:
            return RetrodictionResult(
                known=known,
                computed_value=float("nan"),
                deviation=float("inf"),
                within_tolerance=False,
                details={"error": str(exc)},
            )

        deviation = self._relative_deviation(computed, known.expected_value)
        within = deviation <= known.tolerance

        return RetrodictionResult(
            known=known,
            computed_value=computed,
            deviation=deviation,
            within_tolerance=within,
            details={
                "expected_ratio": known.expected_value,
                "computed_ratio": computed,
            },
        )

    def validate_mup_exponents(
        self,
        compute_fn: Callable[..., float],
    ) -> RetrodictionResult:
        """Check µP learning-rate scaling predictions.

        The compute function should return the empirically measured
        LR scaling exponent with width, expected to be -1 under µP.

        Parameters
        ----------
        compute_fn : callable
            Function(**parameter_regime) -> float.

        Returns
        -------
        result : RetrodictionResult
        """
        known = self._known[2]
        try:
            computed = float(compute_fn(**known.parameter_regime))
        except Exception as exc:
            return RetrodictionResult(
                known=known,
                computed_value=float("nan"),
                deviation=float("inf"),
                within_tolerance=False,
                details={"error": str(exc)},
            )

        deviation = self._relative_deviation(computed, known.expected_value)
        within = deviation <= known.tolerance

        return RetrodictionResult(
            known=known,
            computed_value=computed,
            deviation=deviation,
            within_tolerance=within,
            details={
                "expected_exponent": known.expected_value,
                "computed_exponent": computed,
            },
        )

    def validate_kernel_fixed_point(
        self,
        compute_fn: Callable[..., float],
    ) -> RetrodictionResult:
        """Check infinite-depth kernel fixed point for ReLU.

        The compute function should return the fixed-point value q*
        of the kernel recursion for ReLU with σ_w² = 2, σ_b² = 0.

        Parameters
        ----------
        compute_fn : callable
            Function(**parameter_regime) -> float.

        Returns
        -------
        result : RetrodictionResult
        """
        known = self._known[3]
        try:
            computed = float(compute_fn(**known.parameter_regime))
        except Exception as exc:
            return RetrodictionResult(
                known=known,
                computed_value=float("nan"),
                deviation=float("inf"),
                within_tolerance=False,
                details={"error": str(exc)},
            )

        deviation = self._relative_deviation(computed, known.expected_value)
        within = deviation <= known.tolerance

        return RetrodictionResult(
            known=known,
            computed_value=computed,
            deviation=deviation,
            within_tolerance=within,
            details={
                "expected_fixed_point": known.expected_value,
                "computed_fixed_point": computed,
            },
        )

    # ------------------------------------------------------------------
    # Batch validation
    # ------------------------------------------------------------------

    def run_all(
        self,
        compute_fns: Dict[str, Callable[..., float]],
    ) -> List[RetrodictionResult]:
        """Run all retrodiction validations.

        Parameters
        ----------
        compute_fns : dict
            Mapping from known result name to compute function.
            Expected keys: 'chizat_bach_boundary', 'saxe_linear_dynamics',
            'mup_scaling_exponents', 'kernel_fixed_point'.

        Returns
        -------
        results : list of RetrodictionResult
        """
        dispatch = {
            "chizat_bach_boundary": self.validate_chizat_bach,
            "saxe_linear_dynamics": self.validate_saxe_dynamics,
            "mup_scaling_exponents": self.validate_mup_exponents,
            "kernel_fixed_point": self.validate_kernel_fixed_point,
        }

        results: List[RetrodictionResult] = []
        for name, validator in dispatch.items():
            if name in compute_fns:
                result = validator(compute_fns[name])
                results.append(result)

        return results

    # ------------------------------------------------------------------
    # Analysis and reporting
    # ------------------------------------------------------------------

    def deviation_analysis(
        self, results: List[RetrodictionResult]
    ) -> Dict[str, Any]:
        """Analyze deviations from known results.

        Parameters
        ----------
        results : list of RetrodictionResult

        Returns
        -------
        analysis : dict
            Keys: 'mean_deviation', 'max_deviation', 'pass_rate',
            'per_result' (list of per-result dicts).
        """
        if not results:
            return {
                "mean_deviation": 0.0,
                "max_deviation": 0.0,
                "pass_rate": 0.0,
                "per_result": [],
            }

        deviations = [
            r.deviation for r in results if np.isfinite(r.deviation)
        ]
        passes = sum(1 for r in results if r.within_tolerance)

        per_result = []
        for r in results:
            per_result.append(
                {
                    "name": r.known.name,
                    "deviation": r.deviation,
                    "within_tolerance": r.within_tolerance,
                    "expected": r.known.expected_value,
                    "computed": r.computed_value,
                }
            )

        return {
            "mean_deviation": float(np.mean(deviations)) if deviations else 0.0,
            "max_deviation": float(np.max(deviations)) if deviations else 0.0,
            "pass_rate": passes / len(results) if results else 0.0,
            "per_result": per_result,
        }

    def summary_report(
        self, results: List[RetrodictionResult]
    ) -> str:
        """Format retrodiction results as a human-readable report.

        Parameters
        ----------
        results : list of RetrodictionResult

        Returns
        -------
        report : str
            Formatted report string.
        """
        lines: List[str] = []
        sep = "=" * 72
        lines.append(sep)
        lines.append("RETRODICTION VALIDATION REPORT")
        lines.append(sep)
        lines.append("")

        for r in results:
            status = "PASS" if r.within_tolerance else "FAIL"
            lines.append(f"[{status}] {r.known.name}")
            lines.append(f"  Reference: {r.known.reference}")
            lines.append(f"  Expected:  {r.known.expected_value:.6f}")
            lines.append(f"  Computed:  {r.computed_value:.6f}")
            lines.append(
                f"  Deviation: {r.deviation:.6f} "
                f"(tolerance: {r.known.tolerance:.6f})"
            )
            if r.details:
                for k, v in r.details.items():
                    lines.append(f"  {k}: {v}")
            lines.append("")

        # Summary
        analysis = self.deviation_analysis(results)
        lines.append(sep)
        lines.append("SUMMARY")
        lines.append(
            f"  Pass rate:      {analysis['pass_rate']:.1%} "
            f"({sum(1 for r in results if r.within_tolerance)}/{len(results)})"
        )
        lines.append(
            f"  Mean deviation: {analysis['mean_deviation']:.6f}"
        )
        lines.append(
            f"  Max deviation:  {analysis['max_deviation']:.6f}"
        )
        lines.append(sep)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _relative_deviation(computed: float, expected: float) -> float:
        """Compute relative deviation.

        Parameters
        ----------
        computed : float
        expected : float

        Returns
        -------
        deviation : float
            |computed - expected| / max(|expected|, 1e-10).
        """
        denom = max(abs(expected), 1e-10)
        return abs(computed - expected) / denom
