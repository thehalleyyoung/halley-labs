"""
Calibration Pipeline for Finite-Width NTK Corrections
======================================================

End-to-end pipeline that chains together NTK computation, polynomial
regression in 1/N, bootstrap confidence intervals, and perturbative
validation to produce calibrated correction coefficients

    Θ(N) = Θ⁽⁰⁾ + Θ⁽¹⁾/N + Θ⁽²⁾/N² + …

from empirical NTK measurements at several finite widths.

Pipeline stages
---------------
1. **Measurement** – evaluate the empirical NTK at each (width, seed) pair.
2. **Regression**  – fit the 1/N polynomial via weighted least squares.
3. **Bootstrap**   – seed-resampling bootstrap for confidence intervals.
4. **Validation**  – perturbative validity checks on the estimated
   correction coefficients.
5. **Reporting**   – human-readable diagnostics and plot-ready data.

References
----------
[1] Huang & Yau, "Dynamics of Deep Neural Networks and Neural Tangent
    Hierarchy", ICML 2020.
[2] Dyer & Gur-Ari, "Asymptotics of Wide Networks from Feynman
    Diagrams", ICLR 2020.
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as sp_linalg

from .regression import (
    CalibrationRegression,
    RegressionResult,
    DesignMatrixBuilder,
)
from .bootstrap import (
    BootstrapCI,
    BootstrapResult,
)

logger = logging.getLogger(__name__)

# ======================================================================
#  Configuration
# ======================================================================


@dataclass
class CalibrationConfig:
    """Configuration for the calibration pipeline.

    Attributes
    ----------
    widths : list of int
        Network widths N₁ < N₂ < … < N_K at which to measure the NTK.
    num_seeds : int
        Number of independent parameter initialisations per width.
    max_order : int
        Maximum order p in the 1/N expansion (0 → constant, 1 → linear,
        2 → quadratic).
    confidence_level : float
        Nominal coverage for bootstrap confidence intervals (e.g. 0.95).
    n_bootstrap : int
        Number of bootstrap resamples.
    regularization : float
        Tikhonov regularisation parameter λ ≥ 0 added to the normal
        equations: (AᵀA + λI) θ = Aᵀy.
    fix_theta_0 : bool
        If ``True``, pin Θ⁽⁰⁾ to ``theta_0_known`` instead of
        estimating it from data.
    theta_0_known : ndarray or None
        Known infinite-width NTK, required when ``fix_theta_0`` is
        ``True``.
    convergence_tol : float
        Tolerance for declaring bootstrap convergence.
    verbose : bool
        If ``True``, emit progress messages to stdout.
    """

    widths: List[int] = field(default_factory=lambda: [64, 128, 256, 512, 1024])
    num_seeds: int = 10
    max_order: int = 2
    confidence_level: float = 0.95
    n_bootstrap: int = 1000
    regularization: float = 0.0
    fix_theta_0: bool = False
    theta_0_known: Optional[NDArray[np.floating]] = None
    convergence_tol: float = 1e-3
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.fix_theta_0 and self.theta_0_known is None:
            raise ValueError(
                "theta_0_known must be provided when fix_theta_0 is True"
            )
        if not (0.0 < self.confidence_level < 1.0):
            raise ValueError(
                f"confidence_level must be in (0, 1), got {self.confidence_level}"
            )
        if self.max_order < 0:
            raise ValueError(f"max_order must be >= 0, got {self.max_order}")
        if self.regularization < 0.0:
            raise ValueError(
                f"regularization must be >= 0, got {self.regularization}"
            )


# ======================================================================
#  Result container
# ======================================================================


@dataclass
class CalibrationResult:
    """Collected outputs from a full calibration run.

    Attributes
    ----------
    theta_0 : ndarray
        Estimated (or known) infinite-width NTK, Θ⁽⁰⁾.
    theta_1 : ndarray
        First-order correction coefficient, Θ⁽¹⁾.
    theta_2 : ndarray
        Second-order correction coefficient, Θ⁽²⁾.  Zero array when
        ``max_order < 2``.
    theta_1_ci_lower : ndarray
        Element-wise lower endpoint of the confidence interval for Θ⁽¹⁾.
    theta_1_ci_upper : ndarray
        Element-wise upper endpoint of the confidence interval for Θ⁽¹⁾.
    regression_result : RegressionResult or dict
        Full output from :class:`CalibrationRegression.fit`.
    bootstrap_result : BootstrapResult or dict
        Full output from :class:`BootstrapCI.bootstrap_over_seeds`.
    validity_ratio : float
        ‖Θ⁽¹⁾‖_op / ‖Θ⁽⁰⁾‖_op — a ratio ≪ 1 indicates the
        perturbative expansion is well-controlled.
    confidence_level_str : str
        Human-readable confidence level, e.g. ``"95%"``.
    convergence_info : dict
        Diagnostics on bootstrap convergence (ESS, convergence flag).
    diagnostics : dict
        Assorted regression diagnostics (R², condition number, etc.).
    report_text : str
        Pre-formatted multi-line report string.
    """

    theta_0: NDArray[np.floating]
    theta_1: NDArray[np.floating]
    theta_2: NDArray[np.floating]
    theta_1_ci_lower: NDArray[np.floating]
    theta_1_ci_upper: NDArray[np.floating]
    regression_result: Union[RegressionResult, Dict[str, Any]]
    bootstrap_result: Union[BootstrapResult, Dict[str, Any]]
    validity_ratio: float
    confidence_level_str: str
    convergence_info: Dict[str, Any]
    diagnostics: Dict[str, Any]
    report_text: str


# ======================================================================
#  Input validation helpers
# ======================================================================


class CalibrationValidator:
    """Pre-flight checks on widths and measurements.

    Provides static methods to verify that the experimental design is
    sound before running expensive NTK computations.
    """

    # ------------------------------------------------------------------
    #  Width selection
    # ------------------------------------------------------------------

    @staticmethod
    def validate_width_selection(widths: Sequence[int]) -> Dict[str, Any]:
        """Check that *widths* are sorted, unique, and sufficient.

        Parameters
        ----------
        widths : sequence of int
            Proposed network widths.

        Returns
        -------
        dict
            ``{"valid": bool, "warnings": list[str], "widths_sorted": list[int]}``.
        """
        widths_arr = np.asarray(widths, dtype=int)
        warns: List[str] = []

        if widths_arr.size < 3:
            warns.append(
                f"At least 3 widths are recommended; got {widths_arr.size}."
            )

        if not np.all(widths_arr > 0):
            warns.append("All widths must be positive integers.")

        sorted_widths = np.sort(np.unique(widths_arr))
        if sorted_widths.size < widths_arr.size:
            warns.append("Duplicate widths detected and removed.")

        ratio = sorted_widths[-1] / sorted_widths[0] if sorted_widths.size > 1 else 1
        if ratio < 4:
            warns.append(
                f"Width range ratio N_max/N_min = {ratio:.1f} < 4; "
                "wider spread improves conditioning."
            )

        # Check spacing is roughly geometric (good for 1/N fits)
        if sorted_widths.size >= 3:
            log_widths = np.log(sorted_widths.astype(float))
            gaps = np.diff(log_widths)
            cv = np.std(gaps) / (np.mean(gaps) + 1e-12)
            if cv > 0.5:
                warns.append(
                    "Widths are not approximately geometrically spaced; "
                    "consider using powers of 2 for better conditioning."
                )

        valid = all(
            "must be positive" not in w and "At least 3" not in w for w in warns
        )
        return {
            "valid": valid,
            "warnings": warns,
            "widths_sorted": sorted_widths.tolist(),
        }

    # ------------------------------------------------------------------
    #  Measurement consistency
    # ------------------------------------------------------------------

    @staticmethod
    def validate_measurements(
        ntk_measurements: Dict[int, List[NDArray[np.floating]]],
        widths: Sequence[int],
    ) -> Dict[str, Any]:
        """Check that NTK measurements are consistent across widths.

        Parameters
        ----------
        ntk_measurements : dict[int, list[ndarray]]
            Mapping ``width -> [NTK_seed_1, NTK_seed_2, …]``.
        widths : sequence of int
            Expected widths.

        Returns
        -------
        dict
            ``{"valid": bool, "warnings": list[str], "shapes": dict}``.
        """
        warns: List[str] = []
        shapes: Dict[int, Tuple[int, ...]] = {}

        for w in widths:
            if w not in ntk_measurements:
                warns.append(f"Width {w} missing from measurements.")
                continue
            mats = ntk_measurements[w]
            if len(mats) == 0:
                warns.append(f"Width {w} has zero measurements.")
                continue
            ref_shape = mats[0].shape
            shapes[w] = ref_shape
            for i, m in enumerate(mats):
                if m.shape != ref_shape:
                    warns.append(
                        f"Width {w}, seed {i}: shape {m.shape} != "
                        f"expected {ref_shape}."
                    )
                if not np.all(np.isfinite(m)):
                    warns.append(
                        f"Width {w}, seed {i}: contains non-finite entries."
                    )

        # All shapes should agree
        unique_shapes = set(shapes.values())
        if len(unique_shapes) > 1:
            warns.append(
                f"Inconsistent NTK shapes across widths: {unique_shapes}."
            )

        valid = len(warns) == 0
        return {"valid": valid, "warnings": warns, "shapes": shapes}

    # ------------------------------------------------------------------
    #  Suggest additional widths
    # ------------------------------------------------------------------

    @staticmethod
    def suggest_additional_widths(
        current_widths: Sequence[int],
        regression_result: Union[RegressionResult, Dict[str, Any]],
        n_suggest: int = 3,
    ) -> List[int]:
        """Propose additional widths to improve fit quality.

        Heuristic: add widths that fill the largest gaps in log-space
        and extend the range in both directions.

        Parameters
        ----------
        current_widths : sequence of int
            Already-measured widths.
        regression_result : RegressionResult or dict
            Previous regression output (used for condition analysis).
        n_suggest : int
            Number of widths to suggest.

        Returns
        -------
        list of int
            Suggested new widths, sorted ascending.
        """
        sorted_w = np.sort(np.asarray(current_widths, dtype=float))
        log_w = np.log(sorted_w)
        suggestions: List[int] = []

        # 1. Fill largest gap in log-space
        if len(log_w) >= 2:
            gaps = np.diff(log_w)
            for _ in range(min(n_suggest, len(gaps))):
                idx = int(np.argmax(gaps))
                mid = int(np.round(np.exp(0.5 * (log_w[idx] + log_w[idx + 1]))))
                if mid not in current_widths and mid not in suggestions:
                    suggestions.append(mid)
                gaps[idx] = 0.0  # suppress this gap

        # 2. Extend range: smaller width
        if len(suggestions) < n_suggest:
            smaller = max(int(sorted_w[0] // 2), 4)
            if smaller not in current_widths and smaller not in suggestions:
                suggestions.append(smaller)

        # 3. Extend range: larger width
        if len(suggestions) < n_suggest:
            larger = int(sorted_w[-1] * 2)
            if larger not in current_widths and larger not in suggestions:
                suggestions.append(larger)

        suggestions.sort()
        return suggestions[:n_suggest]


# ======================================================================
#  Main calibration pipeline
# ======================================================================


class CalibrationPipeline:
    """End-to-end calibration of finite-width NTK corrections.

    Orchestrates measurement, regression, bootstrap, and validation
    stages.  Two entry points are provided:

    * :meth:`run` — accepts raw callables and input data; computes NTK
      matrices internally.
    * :meth:`run_from_measurements` — accepts pre-computed NTK matrices
      grouped by seed.

    Parameters
    ----------
    config : CalibrationConfig or None
        Default configuration; can be overridden per-call.
    """

    def __init__(self, config: Optional[CalibrationConfig] = None) -> None:
        self.config = config or CalibrationConfig()

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def run(
        self,
        forward_fn: Callable,
        param_init_fn: Callable,
        X: NDArray[np.floating],
        config: Optional[CalibrationConfig] = None,
    ) -> CalibrationResult:
        """Full calibration pipeline from raw network functions.

        Stages executed:
        1. Compute empirical NTK at each (width, seed) combination.
        2. Fit 1/N polynomial via weighted least squares.
        3. Bootstrap over seeds for confidence intervals on Θ⁽¹⁾.
        4. Validate perturbative consistency.
        5. Assemble report.

        Parameters
        ----------
        forward_fn : callable (params, x) -> output
            Network forward pass.  Must accept a 1-D parameter vector
            and a single input point (or batch).
        param_init_fn : callable (width, seed) -> params
            Returns a 1-D parameter vector for the given width and
            random seed.
        X : ndarray, shape (n, d)
            Input data points at which to evaluate the NTK.
        config : CalibrationConfig or None
            Override for this run only.

        Returns
        -------
        CalibrationResult
        """
        cfg = config or self.config
        t0 = time.time()
        self._log("Starting full calibration pipeline …")

        # --- Validate widths ---
        width_check = CalibrationValidator.validate_width_selection(cfg.widths)
        for w in width_check["warnings"]:
            self._log(f"  [width-check] {w}")
        widths = width_check["widths_sorted"]

        # --- Stage 1: Compute NTK measurements ---
        self._log("Stage 1/5: Computing NTK measurements …")
        ntk_by_seed = self._compute_ntk_measurements(
            forward_fn, param_init_fn, X, widths, cfg.num_seeds
        )

        # --- Stages 2–5 delegated ---
        result = self.run_from_measurements(ntk_by_seed, widths, cfg)

        elapsed = time.time() - t0
        self._log(f"Pipeline complete in {elapsed:.1f}s.")
        return result

    def run_from_measurements(
        self,
        ntk_measurements_by_seed: Dict[int, Dict[int, NDArray[np.floating]]],
        widths: Sequence[int],
        config: Optional[CalibrationConfig] = None,
    ) -> CalibrationResult:
        """Run the pipeline from pre-computed NTK measurements.

        Parameters
        ----------
        ntk_measurements_by_seed : dict[int, dict[int, ndarray]]
            Mapping ``seed -> {width: NTK_matrix}``.
        widths : sequence of int
            Network widths (must match keys in inner dicts).
        config : CalibrationConfig or None
            Override for this run only.

        Returns
        -------
        CalibrationResult
        """
        cfg = config or self.config
        widths = list(widths)

        # Reorganise: width -> [NTK_seed_0, NTK_seed_1, …]
        ntk_by_width: Dict[int, List[NDArray[np.floating]]] = {
            w: [] for w in widths
        }
        for seed_data in ntk_measurements_by_seed.values():
            for w in widths:
                if w in seed_data:
                    ntk_by_width[w].append(seed_data[w])

        # Validate measurements
        meas_check = CalibrationValidator.validate_measurements(
            ntk_by_width, widths
        )
        for w in meas_check["warnings"]:
            self._log(f"  [meas-check] {w}")

        # Mean NTK per width (for regression)
        ntk_means: Dict[int, NDArray[np.floating]] = {
            w: np.mean(np.stack(mats, axis=0), axis=0)
            for w, mats in ntk_by_width.items()
            if len(mats) > 0
        }

        # --- Stage 2: Regression ---
        self._log("Stage 2/5: Regression fit …")
        reg_result, theta_0, theta_1, theta_2 = self._run_regression(
            ntk_means, widths, cfg
        )

        # --- Stage 3: Bootstrap ---
        self._log("Stage 3/5: Bootstrap confidence intervals …")
        boot_result, ci_lower, ci_upper = self._run_bootstrap(
            ntk_measurements_by_seed, widths, cfg
        )

        # --- Stage 4: Validation ---
        self._log("Stage 4/5: Perturbative validation …")
        validity_ratio, convergence_info = self._validate_results(
            theta_0, theta_1, theta_2, cfg
        )

        # --- Optional: compare with known Θ⁽⁰⁾ ---
        if cfg.theta_0_known is not None:
            known_info = self._validate_against_known(theta_0, cfg.theta_0_known)
            convergence_info["known_theta_0_comparison"] = known_info

        # --- Diagnostics dict ---
        diagnostics = self._collect_diagnostics(
            reg_result, boot_result, validity_ratio, widths, cfg
        )

        # --- Stage 5: Report ---
        self._log("Stage 5/5: Generating report …")
        conf_str = f"{int(cfg.confidence_level * 100)}%"

        result = CalibrationResult(
            theta_0=theta_0,
            theta_1=theta_1,
            theta_2=theta_2,
            theta_1_ci_lower=ci_lower,
            theta_1_ci_upper=ci_upper,
            regression_result=reg_result,
            bootstrap_result=boot_result,
            validity_ratio=validity_ratio,
            confidence_level_str=conf_str,
            convergence_info=convergence_info,
            diagnostics=diagnostics,
            report_text="",  # filled below
        )
        result.report_text = self._generate_report(result)
        return result

    # ------------------------------------------------------------------
    #  Stage 1: NTK measurement
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_ntk_measurements(
        forward_fn: Callable,
        param_init_fn: Callable,
        X: NDArray[np.floating],
        widths: Sequence[int],
        num_seeds: int,
    ) -> Dict[int, Dict[int, NDArray[np.floating]]]:
        """Compute empirical NTK at each (width, seed) pair.

        Parameters
        ----------
        forward_fn : callable (params, x) -> output
        param_init_fn : callable (width, seed) -> params
        X : ndarray, shape (n, d)
        widths : sequence of int
        num_seeds : int

        Returns
        -------
        dict[int, dict[int, ndarray]]
            ``seed -> {width: NTK_matrix}``.
        """
        from ..kernel_engine.ntk import EmpiricalNTK

        results: Dict[int, Dict[int, NDArray[np.floating]]] = {}
        for seed in range(num_seeds):
            results[seed] = {}
            for w in widths:
                params = param_init_fn(w, seed)
                ntk_mat = EmpiricalNTK.compute_ntk(forward_fn, params, X)
                results[seed][w] = ntk_mat
        return results

    # ------------------------------------------------------------------
    #  Stage 2: Regression
    # ------------------------------------------------------------------

    def _run_regression(
        self,
        ntk_means: Dict[int, NDArray[np.floating]],
        widths: Sequence[int],
        config: CalibrationConfig,
    ) -> Tuple[
        Union[RegressionResult, Dict[str, Any]],
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
    ]:
        """Fit the 1/N polynomial to width-averaged NTK matrices.

        Returns
        -------
        (regression_result, theta_0, theta_1, theta_2)
        """
        regression = CalibrationRegression()

        # Stack measurements in width order
        sorted_widths = sorted(widths)
        ntk_stack = np.stack(
            [ntk_means[w] for w in sorted_widths], axis=0
        )  # shape (K, d, d) or (K, d)

        # Weights: inverse variance heuristic — wider nets get more weight
        weights = np.array(
            [float(w) for w in sorted_widths], dtype=np.float64
        )
        weights = weights / weights.sum()

        reg_result = regression.fit(
            ntk_stack,
            sorted_widths,
            weights=weights,
        )

        # Extract coefficients from regression result
        if isinstance(reg_result, dict):
            coeffs = reg_result.get("coefficients", reg_result.get("theta"))
        else:
            coeffs = getattr(reg_result, "coefficients", None)
            if coeffs is None:
                coeffs = getattr(reg_result, "theta", None)

        mat_shape = ntk_stack.shape[1:]
        theta_0 = np.zeros(mat_shape, dtype=np.float64)
        theta_1 = np.zeros(mat_shape, dtype=np.float64)
        theta_2 = np.zeros(mat_shape, dtype=np.float64)

        if coeffs is not None:
            # coeffs may be (p+1, d, d) or (p+1, d) — first axis = order
            if coeffs.ndim >= 2:
                theta_0 = coeffs[0]
                if coeffs.shape[0] > 1:
                    theta_1 = coeffs[1]
                if coeffs.shape[0] > 2:
                    theta_2 = coeffs[2]
            else:
                # Scalar / 1-D case: treat as flattened
                theta_0 = coeffs[0:1].reshape(mat_shape) if coeffs.size > 0 else theta_0

        # Override Θ⁽⁰⁾ if fixed
        if config.fix_theta_0 and config.theta_0_known is not None:
            theta_0 = np.asarray(config.theta_0_known, dtype=np.float64)

        return reg_result, theta_0, theta_1, theta_2

    # ------------------------------------------------------------------
    #  Stage 3: Bootstrap
    # ------------------------------------------------------------------

    def _run_bootstrap(
        self,
        ntk_measurements_by_seed: Dict[int, Dict[int, NDArray[np.floating]]],
        widths: Sequence[int],
        config: CalibrationConfig,
    ) -> Tuple[
        Union[BootstrapResult, Dict[str, Any]],
        NDArray[np.floating],
        NDArray[np.floating],
    ]:
        """Compute bootstrap confidence intervals for Θ⁽¹⁾.

        Returns
        -------
        (bootstrap_result, ci_lower, ci_upper)
        """
        bootstrap = BootstrapCI(
            n_bootstrap=config.n_bootstrap,
            confidence_level=config.confidence_level,
        )

        # Build regression function for bootstrap resampling
        def regression_fn(
            ntk_per_width: Dict[int, NDArray[np.floating]],
        ) -> NDArray[np.floating]:
            """Return Θ⁽¹⁾ estimate from a single bootstrap resample."""
            reg = CalibrationRegression()
            sorted_w = sorted(ntk_per_width.keys())
            stack = np.stack([ntk_per_width[w] for w in sorted_w], axis=0)
            result = reg.fit(stack, sorted_w)
            if isinstance(result, dict):
                coeffs = result.get("coefficients", result.get("theta"))
            else:
                coeffs = getattr(result, "coefficients", None)
                if coeffs is None:
                    coeffs = getattr(result, "theta", None)
            if coeffs is not None and coeffs.shape[0] > 1:
                return coeffs[1]  # Θ⁽¹⁾
            return np.zeros_like(stack[0])

        boot_result = bootstrap.bootstrap_over_seeds(
            ntk_measurements_by_seed, widths, regression_fn
        )

        if isinstance(boot_result, dict):
            ci_lower = boot_result.get("ci_lower", np.array([]))
            ci_upper = boot_result.get("ci_upper", np.array([]))
        else:
            ci_lower = boot_result.ci_lower
            ci_upper = boot_result.ci_upper

        return boot_result, ci_lower, ci_upper

    # ------------------------------------------------------------------
    #  Stage 4: Validation
    # ------------------------------------------------------------------

    def _validate_results(
        self,
        theta_0: NDArray[np.floating],
        theta_1: NDArray[np.floating],
        theta_2: NDArray[np.floating],
        config: CalibrationConfig,
    ) -> Tuple[float, Dict[str, Any]]:
        """Run perturbative validity checks.

        Returns
        -------
        (validity_ratio, convergence_info)
            *validity_ratio* is ‖Θ⁽¹⁾‖_op / ‖Θ⁽⁰⁾‖_op.
        """
        try:
            from ..corrections.perturbation import PerturbativeValidator

            validator = PerturbativeValidator()
            validity = validator.validate(theta_0, theta_1, theta_2)
            if isinstance(validity, dict):
                ratio = validity.get("validity_ratio", float("nan"))
            else:
                ratio = getattr(
                    validity,
                    "validity_ratio",
                    getattr(validity, "ratio", float("nan")),
                )
        except (ImportError, Exception) as exc:
            warnings.warn(
                f"PerturbativeValidator unavailable ({exc}); "
                "falling back to manual norm ratio.",
                stacklevel=2,
            )
            ratio = self._manual_validity_ratio(theta_0, theta_1)

        convergence_info = self._assess_convergence(
            theta_0, theta_1, theta_2, config
        )
        convergence_info["validity_ratio"] = ratio
        return ratio, convergence_info

    @staticmethod
    def _manual_validity_ratio(
        theta_0: NDArray[np.floating],
        theta_1: NDArray[np.floating],
    ) -> float:
        """‖Θ⁽¹⁾‖_op / ‖Θ⁽⁰⁾‖_op computed via SVD."""
        if theta_0.ndim >= 2:
            norm_0 = float(sp_linalg.svdvals(theta_0)[0])
            norm_1 = float(sp_linalg.svdvals(theta_1)[0])
        else:
            norm_0 = float(np.linalg.norm(theta_0))
            norm_1 = float(np.linalg.norm(theta_1))
        if norm_0 < 1e-15:
            return float("inf")
        return norm_1 / norm_0

    @staticmethod
    def _assess_convergence(
        theta_0: NDArray[np.floating],
        theta_1: NDArray[np.floating],
        theta_2: NDArray[np.floating],
        config: CalibrationConfig,
    ) -> Dict[str, Any]:
        """Heuristic convergence assessment for the 1/N series.

        Uses the ratio |Θ⁽²⁾| / |Θ⁽¹⁾| as a proxy for the
        convergence rate of the perturbation series.
        """
        norm_1 = float(np.linalg.norm(theta_1))
        norm_2 = float(np.linalg.norm(theta_2))
        if norm_1 > 1e-15:
            ratio_21 = norm_2 / norm_1
        else:
            ratio_21 = 0.0

        converged = ratio_21 < config.convergence_tol or norm_2 < 1e-12
        return {
            "norm_theta_1": norm_1,
            "norm_theta_2": norm_2,
            "ratio_theta2_theta1": ratio_21,
            "series_converged": converged,
            "convergence_tol": config.convergence_tol,
        }

    # ------------------------------------------------------------------
    #  Validate against known Θ⁽⁰⁾
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_against_known(
        theta_0_est: NDArray[np.floating],
        theta_0_known: NDArray[np.floating],
    ) -> Dict[str, Any]:
        """Compare estimated Θ⁽⁰⁾ with the known infinite-width NTK.

        Parameters
        ----------
        theta_0_est : ndarray
            Estimated infinite-width NTK from regression.
        theta_0_known : ndarray
            Ground-truth infinite-width NTK.

        Returns
        -------
        dict
            ``{"relative_error", "frobenius_diff", "max_entry_diff",
            "agreement"}``.
        """
        diff = theta_0_est - theta_0_known
        frob_diff = float(np.linalg.norm(diff))
        frob_known = float(np.linalg.norm(theta_0_known))
        rel_err = frob_diff / max(frob_known, 1e-15)
        max_diff = float(np.max(np.abs(diff)))
        return {
            "relative_error": rel_err,
            "frobenius_diff": frob_diff,
            "max_entry_diff": max_diff,
            "agreement": rel_err < 0.05,
        }

    # ------------------------------------------------------------------
    #  Diagnostics collection
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_diagnostics(
        reg_result: Union[RegressionResult, Dict[str, Any]],
        boot_result: Union[BootstrapResult, Dict[str, Any]],
        validity_ratio: float,
        widths: Sequence[int],
        config: CalibrationConfig,
    ) -> Dict[str, Any]:
        """Collect diagnostics from all stages into a single dict."""
        diag: Dict[str, Any] = {
            "widths": list(widths),
            "num_seeds": config.num_seeds,
            "max_order": config.max_order,
            "n_bootstrap": config.n_bootstrap,
            "regularization": config.regularization,
            "validity_ratio": validity_ratio,
        }

        # Regression diagnostics
        if isinstance(reg_result, dict):
            diag["r_squared"] = reg_result.get("r_squared", None)
            diag["condition_number"] = reg_result.get("condition_number", None)
        else:
            diag["r_squared"] = getattr(reg_result, "r_squared", None)
            diag["condition_number"] = getattr(
                reg_result, "condition_number", None
            )

        # Bootstrap diagnostics
        if isinstance(boot_result, dict):
            diag["bootstrap_converged"] = boot_result.get("converged", None)
        else:
            diag["bootstrap_converged"] = getattr(
                boot_result, "converged", None
            )

        return diag

    # ------------------------------------------------------------------
    #  Stage 5: Reporting
    # ------------------------------------------------------------------

    def _generate_report(self, result: CalibrationResult) -> str:
        """Generate a human-readable calibration report.

        Parameters
        ----------
        result : CalibrationResult
            Completed calibration result.

        Returns
        -------
        str
            Multi-line report.
        """
        lines: List[str] = []
        sep = "=" * 60
        lines.append(sep)
        lines.append("  Finite-Width NTK Calibration Report")
        lines.append(sep)
        lines.append("")

        # --- Coefficient norms ---
        lines.append("Coefficient norms")
        lines.append("-" * 40)
        lines.append(f"  ‖Θ⁽⁰⁾‖_F = {np.linalg.norm(result.theta_0):.6e}")
        lines.append(f"  ‖Θ⁽¹⁾‖_F = {np.linalg.norm(result.theta_1):.6e}")
        lines.append(f"  ‖Θ⁽²⁾‖_F = {np.linalg.norm(result.theta_2):.6e}")
        lines.append("")

        # --- Validity ---
        lines.append("Perturbative validity")
        lines.append("-" * 40)
        lines.append(
            f"  ‖Θ⁽¹⁾‖/‖Θ⁽⁰⁾‖ = {result.validity_ratio:.4f}"
        )
        status = "PASS ✓" if result.validity_ratio < 1.0 else "WARNING ✗"
        lines.append(f"  Status: {status}")
        lines.append("")

        # --- Convergence ---
        lines.append("Series convergence")
        lines.append("-" * 40)
        conv = result.convergence_info
        lines.append(
            f"  |Θ⁽²⁾|/|Θ⁽¹⁾| = "
            f"{conv.get('ratio_theta2_theta1', float('nan')):.4e}"
        )
        lines.append(
            f"  Converged: {conv.get('series_converged', 'unknown')}"
        )
        lines.append("")

        # --- Confidence intervals ---
        lines.append(
            f"Bootstrap {result.confidence_level_str} CI for Θ⁽¹⁾"
        )
        lines.append("-" * 40)
        if result.theta_1_ci_lower.size <= 9:
            lines.append(f"  Lower: {result.theta_1_ci_lower}")
            lines.append(f"  Upper: {result.theta_1_ci_upper}")
        else:
            lines.append(
                f"  Lower (first 3): "
                f"{result.theta_1_ci_lower.ravel()[:3]}"
            )
            lines.append(
                f"  Upper (first 3): "
                f"{result.theta_1_ci_upper.ravel()[:3]}"
            )
        lines.append("")

        # --- Diagnostics ---
        lines.append("Diagnostics")
        lines.append("-" * 40)
        for key, val in result.diagnostics.items():
            lines.append(f"  {key}: {val}")
        lines.append("")

        # --- Known-Θ⁽⁰⁾ comparison ---
        if "known_theta_0_comparison" in conv:
            cmp = conv["known_theta_0_comparison"]
            lines.append("Comparison with known Θ⁽⁰⁾")
            lines.append("-" * 40)
            lines.append(
                f"  Relative error: {cmp['relative_error']:.4e}"
            )
            lines.append(
                f"  Agreement: {'YES' if cmp['agreement'] else 'NO'}"
            )
            lines.append("")

        lines.append(sep)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    #  Convenience helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        """Emit a log message when verbose mode is active."""
        if self.config.verbose:
            logger.info(msg)
            print(msg)

    @staticmethod
    def summary(result: CalibrationResult) -> str:
        """One-line summary of a calibration result.

        Parameters
        ----------
        result : CalibrationResult

        Returns
        -------
        str
        """
        return (
            f"Θ⁽¹⁾/Θ⁽⁰⁾ ratio={result.validity_ratio:.4f}, "
            f"CI=[{np.min(result.theta_1_ci_lower):.3e}, "
            f"{np.max(result.theta_1_ci_upper):.3e}] "
            f"({result.confidence_level_str}), "
            f"converged={result.convergence_info.get('series_converged', '?')}"
        )

    @staticmethod
    def plot_calibration_data(
        result: CalibrationResult,
        ntk_measurements: Dict[int, List[NDArray[np.floating]]],
        widths: Sequence[int],
    ) -> Dict[str, Any]:
        """Return data suitable for plotting the calibration fit.

        Does **not** produce a figure; returns arrays that the caller
        can pass to ``matplotlib`` or any other plotting library.

        Parameters
        ----------
        result : CalibrationResult
        ntk_measurements : dict[int, list[ndarray]]
            ``width -> [NTK matrices]``.
        widths : sequence of int

        Returns
        -------
        dict
            Keys: ``x_inv_widths``, ``y_ntk_norms``, ``y_ntk_means``,
            ``y_ntk_stds``, ``fit_x``, ``fit_y``, ``ci_lower``,
            ``ci_upper``, ``theta_labels``.
        """
        sorted_w = sorted(widths)
        inv_w = np.array([1.0 / w for w in sorted_w])

        # Per-width NTK Frobenius norms
        y_means: List[float] = []
        y_stds: List[float] = []
        y_all: List[NDArray[np.floating]] = []
        for w in sorted_w:
            norms = np.array(
                [float(np.linalg.norm(m)) for m in ntk_measurements[w]]
            )
            y_means.append(float(np.mean(norms)))
            y_stds.append(float(np.std(norms)))
            y_all.append(norms)

        # Dense grid for the fitted curve
        fit_x = np.linspace(0.0, inv_w.max() * 1.1, 200)
        norm_0 = float(np.linalg.norm(result.theta_0))
        norm_1 = float(np.linalg.norm(result.theta_1))
        norm_2 = float(np.linalg.norm(result.theta_2))
        fit_y = norm_0 + norm_1 * fit_x + norm_2 * fit_x ** 2

        # CI band (scaled by norm for plotting)
        ci_lo_norm = float(np.linalg.norm(result.theta_1_ci_lower))
        ci_hi_norm = float(np.linalg.norm(result.theta_1_ci_upper))
        ci_band_lower = norm_0 + ci_lo_norm * fit_x
        ci_band_upper = norm_0 + ci_hi_norm * fit_x

        return {
            "x_inv_widths": inv_w,
            "y_ntk_norms": y_all,
            "y_ntk_means": np.array(y_means),
            "y_ntk_stds": np.array(y_stds),
            "fit_x": fit_x,
            "fit_y": fit_y,
            "ci_lower": ci_band_lower,
            "ci_upper": ci_band_upper,
            "theta_labels": {
                "theta_0_norm": norm_0,
                "theta_1_norm": norm_1,
                "theta_2_norm": norm_2,
            },
        }
