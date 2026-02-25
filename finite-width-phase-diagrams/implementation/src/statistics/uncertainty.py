"""Uncertainty quantification for finite-width phase diagram predictions.

Provides tools for propagating parameter uncertainty through phase diagram
predictions, computing boundary uncertainty via the implicit function theorem,
and decomposing total uncertainty into aleatoric and epistemic components.

Mathematical background
-----------------------
Given calibrated parameters θ with covariance Σ_θ, predictions f(θ) have
approximate covariance

    Σ_f ≈ J Σ_θ Jᵀ

where J = ∂f/∂θ is the Jacobian (linear error propagation / delta method).

For phase boundaries defined implicitly by F(x, θ) = 0, the implicit
function theorem gives

    dx/dθ = −(∂F/∂x)⁻¹ ∂F/∂θ

so that boundary-point covariance is

    Σ_x = (∂F/∂x)⁻¹ (∂F/∂θ) Σ_θ (∂F/∂θ)ᵀ (∂F/∂x)⁻ᵀ.
"""

from __future__ import annotations

import enum
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, stats


# ======================================================================
#  Enumerations
# ======================================================================


class UncertaintySource(enum.Enum):
    """Sources of uncertainty in phase diagram predictions."""

    CALIBRATION = "calibration"
    NYSTROM = "nystrom"
    FINITE_SAMPLE = "finite_sample"
    MODEL_MISMATCH = "model_mismatch"
    NUMERICAL = "numerical"


# ======================================================================
#  Data containers
# ======================================================================


@dataclass
class UncertaintyBudget:
    """Decomposition of total prediction uncertainty by source.

    Parameters
    ----------
    source_contributions : dict
        Mapping from :class:`UncertaintySource` to variance contribution.
    total_variance : float
        Sum of all source variances (assuming independence).
    total_std : float
        Square root of *total_variance*.
    dominant_source : UncertaintySource
        The source contributing the largest share of variance.
    """

    source_contributions: Dict[UncertaintySource, float] = field(
        default_factory=dict
    )
    total_variance: float = 0.0
    total_std: float = 0.0
    dominant_source: UncertaintySource = UncertaintySource.CALIBRATION


@dataclass
class BoundaryUncertainty:
    """Uncertainty of a phase boundary in parameter space.

    Parameters
    ----------
    boundary_points : ndarray
        Boundary locations, shape ``(n_points, dim)``.
    uncertainty_widths : ndarray
        Uncertainty width perpendicular to the boundary at each point,
        shape ``(n_points,)``.
    confidence_level : float
        Confidence level used for the widths.
    covariance_along_boundary : ndarray
        Covariance matrices of boundary-point positions,
        shape ``(n_points, dim, dim)``.
    """

    boundary_points: NDArray[np.floating] = field(
        default_factory=lambda: np.empty((0, 2))
    )
    uncertainty_widths: NDArray[np.floating] = field(
        default_factory=lambda: np.empty(0)
    )
    confidence_level: float = 0.95
    covariance_along_boundary: NDArray[np.floating] = field(
        default_factory=lambda: np.empty((0, 2, 2))
    )


# ======================================================================
#  Main class
# ======================================================================


class UncertaintyQuantifier:
    """Propagate and decompose uncertainty in phase diagram predictions.

    Parameters
    ----------
    confidence_level : float
        Default confidence level for intervals and ellipses (e.g. 0.95).
    """

    def __init__(self, confidence_level: float = 0.95) -> None:
        if not 0.0 < confidence_level < 1.0:
            raise ValueError(
                f"confidence_level must be in (0, 1), got {confidence_level}"
            )
        self.confidence_level = confidence_level

    # ------------------------------------------------------------------
    #  Calibration uncertainty propagation
    # ------------------------------------------------------------------

    def propagate_calibration_uncertainty(
        self,
        calibration_cov: NDArray[np.floating],
        jacobian_fn: Callable[[NDArray[np.floating]], NDArray[np.floating]],
        point: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Linear error propagation: Σ_pred = J Σ_cal Jᵀ.

        Parameters
        ----------
        calibration_cov : ndarray, shape (p, p)
            Covariance matrix of the calibrated parameters.
        jacobian_fn : callable
            Function mapping a parameter vector to the Jacobian matrix
            ``∂(prediction)/∂(parameters)`` with shape ``(m, p)``.
        point : ndarray, shape (p,)
            Parameter vector at which to evaluate the Jacobian.

        Returns
        -------
        ndarray, shape (m, m)
            Predicted covariance matrix.
        """
        calibration_cov = np.asarray(calibration_cov, dtype=np.float64)
        point = np.asarray(point, dtype=np.float64)
        jac = np.atleast_2d(jacobian_fn(point))
        return jac @ calibration_cov @ jac.T

    # ------------------------------------------------------------------
    #  Boundary uncertainty via implicit function theorem
    # ------------------------------------------------------------------

    def boundary_uncertainty_ift(
        self,
        boundary_fn: Callable[
            [NDArray[np.floating], NDArray[np.floating]], NDArray[np.floating]
        ],
        boundary_points: NDArray[np.floating],
        param_cov: NDArray[np.floating],
        theta: Optional[NDArray[np.floating]] = None,
    ) -> BoundaryUncertainty:
        """Boundary uncertainty via the implicit function theorem.

        If *F(x, θ) = 0* defines a phase boundary, then

            dx/dθ = −(∂F/∂x)⁻¹ ∂F/∂θ

        and the covariance of the boundary location is

            Σ_x = G Σ_θ Gᵀ,   G = −(∂F/∂x)⁻¹ (∂F/∂θ).

        Parameters
        ----------
        boundary_fn : callable
            ``F(x, theta)`` returning a scalar (or 1-d array equal in
            length to *x*).  The function must be zero on the boundary.
        boundary_points : ndarray, shape (n_points, dim_x)
            Points lying on the boundary.
        param_cov : ndarray, shape (dim_theta, dim_theta)
            Covariance of the parameters θ.
        theta : ndarray, shape (dim_theta,), optional
            Nominal parameter values.  If *None*, a zero vector of the
            appropriate length is used.

        Returns
        -------
        BoundaryUncertainty
        """
        boundary_points = np.atleast_2d(boundary_points)
        param_cov = np.asarray(param_cov, dtype=np.float64)
        n_points, dim_x = boundary_points.shape
        dim_theta = param_cov.shape[0]

        if theta is None:
            theta = np.zeros(dim_theta, dtype=np.float64)
        theta = np.asarray(theta, dtype=np.float64)

        z_val = stats.norm.ppf(0.5 + self.confidence_level / 2.0)

        covs = np.empty((n_points, dim_x, dim_x))
        widths = np.empty(n_points)

        for i, x in enumerate(boundary_points):
            G = self._implicit_function_jacobian(boundary_fn, x, theta)
            cov_x = G @ param_cov @ G.T
            covs[i] = cov_x
            # Width perpendicular to boundary: largest eigenvalue direction
            eigvals = linalg.eigvalsh(cov_x)
            widths[i] = 2.0 * z_val * np.sqrt(np.max(np.abs(eigvals)))

        return BoundaryUncertainty(
            boundary_points=boundary_points,
            uncertainty_widths=widths,
            confidence_level=self.confidence_level,
            covariance_along_boundary=covs,
        )

    # ------------------------------------------------------------------
    #  Sensitivity analysis
    # ------------------------------------------------------------------

    def sensitivity_analysis(
        self,
        prediction_fn: Callable[[NDArray[np.floating]], NDArray[np.floating]],
        params: NDArray[np.floating],
        param_names: Sequence[str],
        eps: float = 1e-6,
    ) -> dict:
        """Compute sensitivity of predictions to each parameter.

        For each parameter *i* the partial derivative
        ``∂prediction/∂param_i`` is estimated by central finite differences.
        Parameters are then ranked by the L2 norm of their sensitivity
        vector.

        Parameters
        ----------
        prediction_fn : callable
            Maps a parameter vector to a prediction array.
        params : ndarray, shape (p,)
            Nominal parameter vector.
        param_names : sequence of str
            Human-readable name for each parameter.
        eps : float
            Step size for finite differences.

        Returns
        -------
        dict
            Keys: ``"sensitivities"`` (dict name -> ndarray),
            ``"rankings"`` (list of names from most to least sensitive),
            ``"norms"`` (dict name -> float L2 norm).
        """
        params = np.asarray(params, dtype=np.float64)
        jac = self._compute_jacobian_fd(prediction_fn, params, eps=eps)
        # jac has shape (m, p) where p = len(params)

        sensitivities: Dict[str, NDArray[np.floating]] = {}
        norms: Dict[str, float] = {}
        for j, name in enumerate(param_names):
            col = jac[:, j]
            sensitivities[name] = col
            norms[name] = float(np.linalg.norm(col))

        rankings = sorted(norms, key=norms.get, reverse=True)  # type: ignore[arg-type]
        return {
            "sensitivities": sensitivities,
            "rankings": rankings,
            "norms": norms,
        }

    # ------------------------------------------------------------------
    #  Monte Carlo propagation
    # ------------------------------------------------------------------

    def monte_carlo_propagation(
        self,
        prediction_fn: Callable[[NDArray[np.floating]], NDArray[np.floating]],
        param_mean: NDArray[np.floating],
        param_cov: NDArray[np.floating],
        n_samples: int = 1000,
        seed: Optional[int] = None,
    ) -> dict:
        """Propagate parameter uncertainty by Monte Carlo sampling.

        Parameters
        ----------
        prediction_fn : callable
            Maps a parameter vector to a prediction array.
        param_mean : ndarray, shape (p,)
            Mean of the parameter distribution.
        param_cov : ndarray, shape (p, p)
            Covariance of the parameter distribution.
        n_samples : int
            Number of Monte Carlo samples.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        dict
            Keys: ``"mean"`` (ndarray), ``"std"`` (ndarray),
            ``"cov"`` (ndarray), ``"samples"`` (ndarray),
            ``"predictions"`` (ndarray).
        """
        param_mean = np.asarray(param_mean, dtype=np.float64)
        param_cov = np.asarray(param_cov, dtype=np.float64)
        rng = np.random.default_rng(seed)

        samples = rng.multivariate_normal(param_mean, param_cov, size=n_samples)
        predictions = []
        for s in samples:
            predictions.append(np.atleast_1d(prediction_fn(s)))
        predictions_arr = np.array(predictions)

        return {
            "mean": np.mean(predictions_arr, axis=0),
            "std": np.std(predictions_arr, axis=0, ddof=1),
            "cov": np.cov(predictions_arr, rowvar=False),
            "samples": samples,
            "predictions": predictions_arr,
        }

    # ------------------------------------------------------------------
    #  Uncertainty decomposition
    # ------------------------------------------------------------------

    def decompose_uncertainty(self, budget: UncertaintyBudget) -> dict:
        """Separate aleatoric and epistemic uncertainty.

        Aleatoric (irreducible) sources: FINITE_SAMPLE, NUMERICAL.
        Epistemic (reducible) sources: CALIBRATION, MODEL_MISMATCH, NYSTROM.

        Parameters
        ----------
        budget : UncertaintyBudget
            An uncertainty budget with per-source contributions.

        Returns
        -------
        dict
            Keys: ``"aleatoric_variance"``, ``"epistemic_variance"``,
            ``"aleatoric_fraction"``, ``"epistemic_fraction"``,
            ``"aleatoric_sources"`` (dict), ``"epistemic_sources"`` (dict).
        """
        aleatoric_sources = {
            UncertaintySource.FINITE_SAMPLE,
            UncertaintySource.NUMERICAL,
        }
        epistemic_sources = {
            UncertaintySource.CALIBRATION,
            UncertaintySource.MODEL_MISMATCH,
            UncertaintySource.NYSTROM,
        }

        aleatoric_dict: Dict[UncertaintySource, float] = {}
        epistemic_dict: Dict[UncertaintySource, float] = {}
        aleatoric_var = 0.0
        epistemic_var = 0.0

        for src, var in budget.source_contributions.items():
            if src in aleatoric_sources:
                aleatoric_var += var
                aleatoric_dict[src] = var
            elif src in epistemic_sources:
                epistemic_var += var
                epistemic_dict[src] = var

        total = aleatoric_var + epistemic_var
        if total == 0.0:
            total = 1.0  # avoid division by zero

        return {
            "aleatoric_variance": aleatoric_var,
            "epistemic_variance": epistemic_var,
            "aleatoric_fraction": aleatoric_var / total,
            "epistemic_fraction": epistemic_var / total,
            "aleatoric_sources": aleatoric_dict,
            "epistemic_sources": epistemic_dict,
        }

    # ------------------------------------------------------------------
    #  Combining independent uncertainty budgets
    # ------------------------------------------------------------------

    def combine_uncertainties(
        self, budgets: List[UncertaintyBudget]
    ) -> UncertaintyBudget:
        """Combine independent uncertainty budgets by summing variances.

        Parameters
        ----------
        budgets : list of UncertaintyBudget
            Independent uncertainty budgets to combine.

        Returns
        -------
        UncertaintyBudget
            Combined budget with summed source contributions.
        """
        combined: Dict[UncertaintySource, float] = {}
        for b in budgets:
            for src, var in b.source_contributions.items():
                combined[src] = combined.get(src, 0.0) + var

        total_var = sum(combined.values())
        dominant = max(combined, key=combined.get) if combined else UncertaintySource.CALIBRATION  # type: ignore[arg-type]

        return UncertaintyBudget(
            source_contributions=combined,
            total_variance=total_var,
            total_std=np.sqrt(total_var),
            dominant_source=dominant,
        )

    # ------------------------------------------------------------------
    #  Confidence ellipse
    # ------------------------------------------------------------------

    def confidence_ellipse(
        self,
        mean: NDArray[np.floating],
        cov: NDArray[np.floating],
        confidence_level: Optional[float] = None,
        n_points: int = 100,
    ) -> NDArray[np.floating]:
        """Compute points on a 2-D confidence ellipse.

        The ellipse is the set {x : (x−μ)ᵀ Σ⁻¹ (x−μ) ≤ χ²₂(α)} where
        α is the confidence level.

        Parameters
        ----------
        mean : ndarray, shape (2,)
            Centre of the ellipse.
        cov : ndarray, shape (2, 2)
            Covariance matrix.
        confidence_level : float, optional
            Override the instance default.
        n_points : int
            Number of points on the ellipse perimeter.

        Returns
        -------
        ndarray, shape (n_points, 2)
            Points tracing the ellipse boundary.
        """
        mean = np.asarray(mean, dtype=np.float64)
        cov = np.asarray(cov, dtype=np.float64)
        if mean.shape != (2,) or cov.shape != (2, 2):
            raise ValueError("confidence_ellipse requires 2-D mean and cov")

        if confidence_level is None:
            confidence_level = self.confidence_level

        chi2_val = stats.chi2.ppf(confidence_level, df=2)
        eigvals, eigvecs = linalg.eigh(cov)
        # Clamp small negative eigenvalues from numerical noise
        eigvals = np.maximum(eigvals, 0.0)

        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        unit_circle = np.column_stack([np.cos(theta), np.sin(theta)])
        scaled = unit_circle * np.sqrt(eigvals * chi2_val)
        ellipse = scaled @ eigvecs.T + mean
        return ellipse

    # ------------------------------------------------------------------
    #  Prediction interval
    # ------------------------------------------------------------------

    def prediction_interval(
        self,
        mean: float,
        std: float,
        confidence_level: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Compute a symmetric prediction interval assuming normality.

        Parameters
        ----------
        mean : float
            Point prediction.
        std : float
            Standard deviation of the prediction.
        confidence_level : float, optional
            Override the instance default.

        Returns
        -------
        tuple of float
            ``(lower, upper)`` bounds of the interval.
        """
        if confidence_level is None:
            confidence_level = self.confidence_level

        z = stats.norm.ppf(0.5 + confidence_level / 2.0)
        return (mean - z * std, mean + z * std)

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_jacobian_fd(
        fn: Callable[[NDArray[np.floating]], NDArray[np.floating]],
        x: NDArray[np.floating],
        eps: float = 1e-6,
    ) -> NDArray[np.floating]:
        """Finite-difference Jacobian via central differences.

        Parameters
        ----------
        fn : callable
            Function mapping an (p,) vector to an (m,) vector.
        x : ndarray, shape (p,)
            Point at which to evaluate.
        eps : float
            Perturbation size.

        Returns
        -------
        ndarray, shape (m, p)
            Jacobian matrix.
        """
        x = np.asarray(x, dtype=np.float64)
        f0 = np.atleast_1d(fn(x))
        n = len(x)
        m = len(f0)
        jac = np.empty((m, n), dtype=np.float64)

        for j in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[j] += eps
            x_minus[j] -= eps
            jac[:, j] = (np.atleast_1d(fn(x_plus)) - np.atleast_1d(fn(x_minus))) / (
                2.0 * eps
            )

        return jac

    @staticmethod
    def _implicit_function_jacobian(
        F: Callable[
            [NDArray[np.floating], NDArray[np.floating]], NDArray[np.floating]
        ],
        x: NDArray[np.floating],
        theta: NDArray[np.floating],
        eps: float = 1e-6,
    ) -> NDArray[np.floating]:
        """Jacobian dx/dθ via the implicit function theorem.

        Given F(x, θ) = 0:

            dx/dθ = −(∂F/∂x)⁻¹ (∂F/∂θ)

        Partial derivatives are estimated by central finite differences.

        Parameters
        ----------
        F : callable
            ``F(x, theta)`` returning an array with the same length as *x*.
        x : ndarray, shape (dim_x,)
            Point on the boundary.
        theta : ndarray, shape (dim_theta,)
            Parameter values.
        eps : float
            Perturbation size.

        Returns
        -------
        ndarray, shape (dim_x, dim_theta)
            Jacobian G = dx/dθ.
        """
        x = np.asarray(x, dtype=np.float64)
        theta = np.asarray(theta, dtype=np.float64)
        f0 = np.atleast_1d(F(x, theta))
        dim_x = len(x)
        dim_theta = len(theta)
        dim_f = len(f0)

        # ∂F/∂x  (dim_f × dim_x)
        dF_dx = np.empty((dim_f, dim_x), dtype=np.float64)
        for j in range(dim_x):
            xp = x.copy()
            xm = x.copy()
            xp[j] += eps
            xm[j] -= eps
            dF_dx[:, j] = (
                np.atleast_1d(F(xp, theta)) - np.atleast_1d(F(xm, theta))
            ) / (2.0 * eps)

        # ∂F/∂θ  (dim_f × dim_theta)
        dF_dtheta = np.empty((dim_f, dim_theta), dtype=np.float64)
        for j in range(dim_theta):
            tp = theta.copy()
            tm = theta.copy()
            tp[j] += eps
            tm[j] -= eps
            dF_dtheta[:, j] = (
                np.atleast_1d(F(x, tp)) - np.atleast_1d(F(x, tm))
            ) / (2.0 * eps)

        # G = −(∂F/∂x)⁻¹ (∂F/∂θ)
        try:
            G = -linalg.solve(dF_dx, dF_dtheta)
        except linalg.LinAlgError:
            warnings.warn(
                "Singular ∂F/∂x encountered; falling back to least-squares.",
                stacklevel=2,
            )
            G = -np.linalg.lstsq(dF_dx, dF_dtheta, rcond=None)[0]

        return G
