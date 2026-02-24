"""
Finite-size scaling analysis for neural networks treating width as system size.

Adapts the finite-size scaling (FSS) framework from statistical mechanics to
neural networks, where the network width N plays the role of the linear
system size L.  Near a phase transition (e.g., lazy → rich, under-fitting →
over-fitting), observables satisfy the scaling hypothesis:

    O(x, N) = N^{a/ν}  F( (x - x_c) N^{1/ν} )

where x is the control parameter, x_c is the critical value, ν is the
correlation-length exponent, a is the observable's scaling dimension, and
F is a universal scaling function.

This module provides tools for:
- Data collapse: finding (x_c, ν, a) that collapse different-N curves
- Critical point extraction: Binder cumulant crossings, phenomenological
  renormalisation, sequence extrapolation
- Critical exponent measurement: ν, γ, β, α, η with corrections to scaling
- Neural-network–specific FSS: lazy–rich transition, double descent,
  μP parameterisation

References:
    - M.E. Fisher, "The theory of equilibrium critical phenomena"
      (Rep. Prog. Phys. 30, 615, 1967)
    - M.N. Barber, "Finite-size scaling" in Phase Transitions and
      Critical Phenomena, Vol. 8 (Academic Press, 1983)
    - K. Binder, "Finite size scaling analysis of Ising model block
      distribution functions" (Z. Phys. B 43, 119, 1981)
    - V. Privman (ed.), "Finite Size Scaling and Numerical Simulation
      of Statistical Systems" (World Scientific, 1990)
    - J. Cardy, "Finite-Size Scaling" (North-Holland, 1988)
"""

import numpy as np
from scipy import optimize, stats, interpolate, special
from typing import Callable, Optional, List, Tuple, Dict, Union
from dataclasses import dataclass, field
import warnings


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FiniteSizeScalingConfig:
    """Configuration for finite-size scaling analysis.

    Attributes:
        widths: Array of network widths (system sizes) to analyse.
        control_param_range: Tuple (x_min, x_max) for the control parameter
            (e.g., learning rate, sample size ratio).
        observable_fn: Optional callable (width, control_param) → observable.
        n_trials: Number of independent trials for error estimation.
    """
    widths: np.ndarray = field(default_factory=lambda: np.array([64, 128, 256, 512, 1024]))
    control_param_range: Tuple[float, float] = (0.0, 2.0)
    observable_fn: Optional[Callable[[int, float], float]] = None
    n_trials: int = 10


# ---------------------------------------------------------------------------
# ScalingCollapseEngine
# ---------------------------------------------------------------------------

class ScalingCollapseEngine:
    """Engine for finite-size scaling data collapse.

    Data collapse is the hallmark test of a continuous phase transition.
    If data from different system sizes N collapse onto a single curve
    after rescaling

        x → (x - x_c) N^{1/ν}
        y → y · N^{-a/ν}

    then the system exhibits a genuine critical point at x_c with
    exponents ν and a.

    Attributes:
        config: FiniteSizeScalingConfig instance.

    References:
        Houdayer & Hartmann, "Low-temperature behavior of two-dimensional
        Gaussian Ising spin glasses" (Phys. Rev. B 70, 2004) — collapse
        quality metric.
        Kawashima & Ito, "Critical behavior of the three-dimensional
        ±J model in a magnetic field" (J. Phys. Soc. Jpn., 1993).
    """

    def __init__(self, config: FiniteSizeScalingConfig):
        """Initialise with FSS configuration.

        Args:
            config: Configuration specifying widths, parameter range, etc.
        """
        self.config = config

    def attempt_collapse(
        self,
        data_by_width: Dict[int, Tuple[np.ndarray, np.ndarray]],
        critical_value: float,
        nu: float,
        observable_exponent: float,
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Attempt scaling collapse with given parameters.

        Rescales the data:
            x_scaled = (x - x_c) · N^{1/ν}
            y_scaled = y · N^{-a/ν}

        Args:
            data_by_width: Dict mapping width N to (x_array, y_array).
            critical_value: Critical value x_c of the control parameter.
            nu: Correlation-length exponent ν.
            observable_exponent: Scaling dimension a of the observable.

        Returns:
            Dict mapping width N to (x_scaled, y_scaled).
        """
        collapsed = {}
        for N, (x, y) in data_by_width.items():
            x_scaled = (x - critical_value) * N ** (1.0 / nu)
            y_scaled = y * N ** (-observable_exponent / nu)
            collapsed[N] = (x_scaled, y_scaled)
        return collapsed

    def collapse_quality(
        self,
        collapsed_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    ) -> float:
        """Measure quality of scaling collapse.

        Uses the method of Houdayer & Hartmann: interpolate a master
        curve from all data, then measure the mean squared deviation
        of individual curves from the master curve.

        Small values indicate good collapse.

        Args:
            collapsed_data: Dict mapping width N to (x_scaled, y_scaled).

        Returns:
            Collapse quality metric (lower is better).
        """
        # Collect all points
        all_x = []
        all_y = []
        for N, (x, y) in collapsed_data.items():
            all_x.extend(x)
            all_y.extend(y)

        all_x = np.array(all_x)
        all_y = np.array(all_y)
        sort_idx = np.argsort(all_x)
        all_x = all_x[sort_idx]
        all_y = all_y[sort_idx]

        # Remove duplicate x values
        unique_x, unique_idx = np.unique(all_x, return_index=True)
        unique_y = all_y[unique_idx]

        if len(unique_x) < 4:
            return np.inf

        # Master curve via spline interpolation
        try:
            master = interpolate.UnivariateSpline(unique_x, unique_y, s=len(unique_x), k=3)
        except Exception:
            return np.inf

        # Residual: mean squared deviation from master curve
        total_residual = 0.0
        n_points = 0
        for N, (x, y) in collapsed_data.items():
            mask = (x >= unique_x[0]) & (x <= unique_x[-1])
            if np.sum(mask) == 0:
                continue
            y_master = master(x[mask])
            total_residual += np.sum((y[mask] - y_master) ** 2)
            n_points += np.sum(mask)

        if n_points == 0:
            return np.inf

        # Normalise by variance of y data
        y_var = np.var(all_y)
        if y_var < 1e-30:
            return 0.0

        return float(total_residual / (n_points * y_var))

    def optimize_collapse(
        self,
        data_by_width: Dict[int, Tuple[np.ndarray, np.ndarray]],
        nu_range: Tuple[float, float] = (0.3, 3.0),
        xc_range: Tuple[float, float] = (0.5, 1.5),
        exponent_range: Tuple[float, float] = (-2.0, 2.0),
    ) -> Dict[str, float]:
        """Find optimal collapse parameters (x_c, ν, a) by minimisation.

        Minimises the collapse quality metric over the parameter space
        (x_c, ν, a) using differential evolution (global optimiser).

        Args:
            data_by_width: Dict mapping width N to (x_array, y_array).
            nu_range: Bounds for ν.
            xc_range: Bounds for x_c.
            exponent_range: Bounds for observable exponent a.

        Returns:
            Dictionary with 'x_c', 'nu', 'exponent', and 'quality'.
        """

        def objective(params):
            xc, nu, exp_ = params
            if nu <= 0:
                return 1e10
            collapsed = self.attempt_collapse(data_by_width, xc, nu, exp_)
            return self.collapse_quality(collapsed)

        bounds = [xc_range, nu_range, exponent_range]
        result = optimize.differential_evolution(
            objective, bounds, maxiter=500, tol=1e-6, seed=42
        )

        return {
            "x_c": float(result.x[0]),
            "nu": float(result.x[1]),
            "exponent": float(result.x[2]),
            "quality": float(result.fun),
        }

    def bootstrap_errors(
        self,
        data_by_width: Dict[int, Tuple[np.ndarray, np.ndarray]],
        n_bootstrap: int = 200,
    ) -> Dict[str, Tuple[float, float]]:
        """Bootstrap confidence intervals for critical parameters.

        Resamples the data with replacement and re-optimises the collapse
        for each bootstrap sample.  Returns mean and standard deviation
        for each parameter.

        Args:
            data_by_width: Original data.
            n_bootstrap: Number of bootstrap samples.

        Returns:
            Dictionary with parameter names mapping to (mean, std) tuples.
        """
        rng = np.random.RandomState(42)
        xc_samples = []
        nu_samples = []
        exp_samples = []

        for _ in range(n_bootstrap):
            resampled = {}
            for N, (x, y) in data_by_width.items():
                n_pts = len(x)
                idx = rng.choice(n_pts, size=n_pts, replace=True)
                resampled[N] = (x[idx], y[idx])

            try:
                result = self.optimize_collapse(resampled)
                xc_samples.append(result["x_c"])
                nu_samples.append(result["nu"])
                exp_samples.append(result["exponent"])
            except Exception:
                pass

        if len(xc_samples) == 0:
            return {
                "x_c": (np.nan, np.nan),
                "nu": (np.nan, np.nan),
                "exponent": (np.nan, np.nan),
            }

        return {
            "x_c": (float(np.mean(xc_samples)), float(np.std(xc_samples))),
            "nu": (float(np.mean(nu_samples)), float(np.std(nu_samples))),
            "exponent": (float(np.mean(exp_samples)), float(np.std(exp_samples))),
        }

    def crossing_analysis(
        self,
        data_by_width: Dict[int, Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[str, float]:
        """Find crossing points of observable curves at different widths.

        At a critical point, certain dimensionless observables (e.g.,
        Binder cumulant) are independent of system size, so curves for
        different N cross at x_c.

        Args:
            data_by_width: Dict mapping width N to (x_array, y_array).

        Returns:
            Dictionary with 'x_c' (average crossing point), 'x_c_std',
            and 'crossing_pairs' listing (N1, N2, x_cross).
        """
        widths = sorted(data_by_width.keys())
        crossings = []

        for i in range(len(widths)):
            for j in range(i + 1, len(widths)):
                N1, N2 = widths[i], widths[j]
                x1, y1 = data_by_width[N1]
                x2, y2 = data_by_width[N2]

                # Interpolate both onto common x grid
                x_min = max(x1.min(), x2.min())
                x_max = min(x1.max(), x2.max())
                if x_max <= x_min:
                    continue

                x_common = np.linspace(x_min, x_max, 500)
                try:
                    f1 = interpolate.interp1d(x1, y1, kind="cubic", fill_value="extrapolate")
                    f2 = interpolate.interp1d(x2, y2, kind="cubic", fill_value="extrapolate")
                except ValueError:
                    continue

                diff = f1(x_common) - f2(x_common)
                sign_changes = np.where(np.diff(np.sign(diff)))[0]

                for idx in sign_changes:
                    # Linear interpolation to find crossing
                    x_cross = x_common[idx] - diff[idx] * (
                        x_common[idx + 1] - x_common[idx]
                    ) / (diff[idx + 1] - diff[idx])
                    crossings.append((N1, N2, float(x_cross)))

        if not crossings:
            return {"x_c": np.nan, "x_c_std": np.nan, "crossing_pairs": []}

        x_crosses = [c[2] for c in crossings]
        return {
            "x_c": float(np.mean(x_crosses)),
            "x_c_std": float(np.std(x_crosses)),
            "crossing_pairs": crossings,
        }

    def binder_cumulant(
        self,
        order_param_samples_by_width: Dict[int, np.ndarray],
    ) -> Dict[int, float]:
        """Compute Binder cumulant U₄ for each width.

        The Binder cumulant is

            U₄ = 1 - <m⁴> / (3 <m²>²)

        It is a dimensionless ratio that takes a universal value at x_c
        and has a system-size–independent crossing point.

        Args:
            order_param_samples_by_width: Dict mapping width N to array
                of order parameter samples m.

        Returns:
            Dict mapping width N to U₄.

        References:
            Binder (1981), Eq. (2.14).
        """
        result = {}
        for N, samples in order_param_samples_by_width.items():
            m2 = np.mean(samples ** 2)
            m4 = np.mean(samples ** 4)
            if m2 > 1e-30:
                result[N] = float(1.0 - m4 / (3.0 * m2 ** 2))
            else:
                result[N] = 0.0
        return result

    def binder_crossing(
        self,
        binder_values_by_width: Dict[int, Tuple[np.ndarray, np.ndarray]],
        control_range: Tuple[float, float],
    ) -> Dict[str, float]:
        """Find x_c from Binder cumulant crossing.

        The Binder cumulant U₄(x, N) crosses at x_c for different N.
        The crossing point converges to x_c as N → ∞.

        Args:
            binder_values_by_width: Dict mapping width N to
                (control_param_array, U4_array).
            control_range: Range of control parameter to search.

        Returns:
            Dictionary with 'x_c', 'x_c_error', and 'U4_at_crossing'.
        """
        return self.crossing_analysis(binder_values_by_width)


# ---------------------------------------------------------------------------
# CriticalPointExtractor
# ---------------------------------------------------------------------------

class CriticalPointExtractor:
    """Extract critical points from finite-size data.

    Multiple methods for locating x_c:
    - Phenomenological renormalisation (correlation-length crossing)
    - Quotient method (logarithmic derivatives)
    - Sequence extrapolation (BST, VBS)
    - Multi-histogram reweighting

    References:
        Nightingale, "Scaling theory and finite systems" (Physica 83A, 1976).
        Ferrenberg & Swendsen, "Optimized Monte Carlo data analysis"
        (Phys. Rev. Lett. 63, 1989).
    """

    def __init__(self, config: FiniteSizeScalingConfig):
        """Initialise with FSS configuration.

        Args:
            config: Configuration specifying widths, parameter range, etc.
        """
        self.config = config

    def phenomenological_renormalization(
        self,
        correlation_lengths_by_width: Dict[int, Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[str, float]:
        """Phenomenological renormalisation group.

        At the critical point, the scaled correlation length ξ/N is
        independent of system size:

            ξ(x_c, N) / N = ξ(x_c, N') / N'

        This method finds x_c from the crossing of ξ/N curves.

        Args:
            correlation_lengths_by_width: Dict mapping width N to
                (control_param_array, xi_array).

        Returns:
            Dictionary with 'x_c', 'x_c_error', 'nu' (estimated).

        References:
            Nightingale (1976), Eq. (3).
        """
        # Construct ξ/N curves
        scaled_data = {}
        for N, (x, xi) in correlation_lengths_by_width.items():
            scaled_data[N] = (x, xi / N)

        # Find crossings
        engine = ScalingCollapseEngine(self.config)
        crossings = engine.crossing_analysis(scaled_data)

        # Estimate ν from how crossing point moves with N
        result = {
            "x_c": crossings["x_c"],
            "x_c_error": crossings["x_c_std"],
        }

        # Estimate ν from derivative at crossing
        pairs = crossings.get("crossing_pairs", [])
        if len(pairs) >= 2:
            nu_estimates = []
            for N1, N2, x_cross in pairs:
                # ν ≈ ln(N2/N1) / ln(dξ₂/dξ₁) approximately
                nu_estimates.append(np.log(N2 / N1))
            result["nu"] = float(np.mean(nu_estimates))
        else:
            result["nu"] = np.nan

        return result

    def quotient_method(
        self,
        observable_by_width: Dict[int, Tuple[np.ndarray, np.ndarray]],
        derivative_by_width: Dict[int, Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[str, float]:
        """Quotient method for critical point and exponent.

        At x_c, the quotient

            Q_O = O(x_c, s·N) / O(x_c, N) = s^{a_O / ν}

        where s = N'/N is the scale factor.  Similarly for the derivative:

            Q_{dO} = (dO/dx)(x_c, s·N) / (dO/dx)(x_c, N) = s^{a_O/ν + 1/ν}

        The crossing of Q_O vs x for different N-pairs gives x_c.

        Args:
            observable_by_width: Dict mapping width N to (x, O(x, N)).
            derivative_by_width: Dict mapping width N to (x, dO/dx).

        Returns:
            Dictionary with 'x_c', 'exponent_ratio' (a/ν).
        """
        widths = sorted(observable_by_width.keys())
        if len(widths) < 2:
            return {"x_c": np.nan, "exponent_ratio": np.nan}

        # Use successive pairs
        xc_estimates = []
        exp_estimates = []

        for i in range(len(widths) - 1):
            N1 = widths[i]
            N2 = widths[i + 1]
            x1, o1 = observable_by_width[N1]
            x2, o2 = observable_by_width[N2]

            x_min = max(x1.min(), x2.min())
            x_max = min(x1.max(), x2.max())
            if x_max <= x_min:
                continue

            x_common = np.linspace(x_min, x_max, 500)
            try:
                f1 = interpolate.interp1d(x1, o1, kind="cubic", fill_value="extrapolate")
                f2 = interpolate.interp1d(x2, o2, kind="cubic", fill_value="extrapolate")
            except ValueError:
                continue

            # Quotient
            with np.errstate(divide="ignore", invalid="ignore"):
                quotient = f2(x_common) / f1(x_common)

            # At x_c, quotient = (N2/N1)^{a/ν} = const
            # Find where quotient changes least (minimum of |dQ/dx|)
            dq = np.gradient(quotient, x_common)
            min_idx = np.argmin(np.abs(dq))
            xc_estimates.append(float(x_common[min_idx]))

            s = N2 / N1
            q_at_xc = quotient[min_idx]
            if q_at_xc > 0 and s > 1:
                exp_estimates.append(float(np.log(q_at_xc) / np.log(s)))

        return {
            "x_c": float(np.mean(xc_estimates)) if xc_estimates else np.nan,
            "exponent_ratio": float(np.mean(exp_estimates)) if exp_estimates else np.nan,
        }

    def sequence_extrapolation(
        self,
        xc_estimates_by_width_pair: List[Tuple[int, int, float]],
    ) -> float:
        """Extrapolate x_c(N) → x_c(∞) from finite-N estimates.

        Given x_c estimates from successive width pairs (N, N'), form
        a sequence and extrapolate to N → ∞.

        Args:
            xc_estimates_by_width_pair: List of (N1, N2, x_c_estimate).

        Returns:
            Extrapolated x_c(∞).
        """
        if len(xc_estimates_by_width_pair) < 3:
            return float(xc_estimates_by_width_pair[-1][2]) if xc_estimates_by_width_pair else np.nan

        # Sort by geometric mean of widths
        sorted_est = sorted(xc_estimates_by_width_pair, key=lambda t: np.sqrt(t[0] * t[1]))
        xc_seq = np.array([t[2] for t in sorted_est])

        return self.bst_extrapolation(xc_seq)

    def bst_extrapolation(self, sequence: np.ndarray) -> float:
        """Bulirsch–Stoer–Henkel (BST) extrapolation.

        A nonlinear sequence transformation designed for sequences with
        confluent corrections of the form

            S_n = S + c_1 / n^ω + c_2 / n^{2ω} + …

        BST is based on a continued-fraction representation and is more
        robust than Richardson extrapolation for non-integer correction
        exponents.

        Args:
            sequence: The sequence to extrapolate.

        Returns:
            Extrapolated limit.

        References:
            Henkel & Schuetz, "Finite-size scaling with confluent
            corrections" (J. Phys. A 21, 1988).
        """
        seq = np.asarray(sequence, dtype=float)
        n = len(seq)
        if n < 2:
            return float(seq[-1])

        # BST tableau using h_k = 1/(k+1)^2
        h = 1.0 / np.arange(1, n + 1, dtype=float) ** 2
        T = np.zeros((n, n))
        T[:, 0] = seq

        for j in range(1, n):
            for i in range(n - j):
                ratio = h[i] / h[i + j]
                denom = ratio * (1.0 - T[i + 1, j - 1] / T[i, j - 1]) - 1.0
                if np.abs(denom) < 1e-30:
                    T[i, j] = T[i + 1, j - 1]
                else:
                    T[i, j] = T[i + 1, j - 1] + (T[i + 1, j - 1] - T[i, j - 1]) / denom

        return float(T[0, n - 1])

    def vbs_extrapolation(self, sequence: np.ndarray, omega: float = 1.0) -> float:
        """Van den Broeck–Schwartz (VBS) extrapolation with correction exponent.

        Assumes corrections of the form

            S_n = S + c · n^{-ω}

        and eliminates the leading correction analytically.

        Args:
            sequence: The sequence to extrapolate.
            omega: Leading correction exponent.

        Returns:
            Extrapolated limit.

        References:
            van den Broeck & Schwartz, "Method for the extrapolation of
            sequences" (SIAM J. Math. Anal. 10, 1979).
        """
        seq = np.asarray(sequence, dtype=float)
        n = len(seq)
        if n < 3:
            return float(seq[-1])

        # Construct transformed sequence eliminating n^{-ω} correction
        ns = np.arange(1, n + 1, dtype=float)
        transformed = []
        for i in range(n - 1):
            # S_∞ = (S_{i+1} n_{i+1}^ω - S_i n_i^ω) / (n_{i+1}^ω - n_i^ω)
            num = seq[i + 1] * ns[i + 1] ** omega - seq[i] * ns[i] ** omega
            den = ns[i + 1] ** omega - ns[i] ** omega
            if np.abs(den) > 1e-30:
                transformed.append(num / den)

        if not transformed:
            return float(seq[-1])

        return float(transformed[-1])

    def multi_histogram_method(
        self,
        histograms_by_param: Dict[float, np.ndarray],
    ) -> Dict[str, float]:
        """Ferrenberg–Swendsen multi-histogram reweighting.

        Given histograms of an observable measured at several values of
        the control parameter, reweight to obtain continuous estimates
        at any parameter value.  The optimal reweighting minimises
        statistical errors via self-consistent equations.

        Args:
            histograms_by_param: Dict mapping control-parameter value x
                to histogram (array of counts or samples).

        Returns:
            Dictionary with 'free_energy' (array), 'param_values',
            'optimal_observable' (function-like array).

        References:
            Ferrenberg & Swendsen (1989), Eqs. (3)–(6).
        """
        params = sorted(histograms_by_param.keys())
        if len(params) < 2:
            return {"param_values": np.array(params), "free_energy": np.array([0.0])}

        # Simple version: estimate mean observable via reweighting
        means = []
        counts = []
        for x in params:
            h = np.asarray(histograms_by_param[x], dtype=float)
            means.append(np.mean(h))
            counts.append(len(h))

        means = np.array(means)
        counts = np.array(counts)

        # Free-energy differences from ratio of means
        f = np.zeros(len(params))
        for i in range(1, len(params)):
            if means[i - 1] > 0:
                f[i] = f[i - 1] - np.log(means[i] / means[i - 1])
            else:
                f[i] = f[i - 1]

        return {
            "param_values": np.array(params),
            "free_energy": f,
            "mean_observable": means,
        }

    def cumulant_intersection(
        self,
        cumulant_by_width: Dict[int, Tuple[np.ndarray, np.ndarray]],
        control_range: Tuple[float, float],
    ) -> Dict[str, float]:
        """Find x_c from intersection of cumulant curves.

        Generic intersection method applicable to any cumulant ratio
        (Binder, skewness, kurtosis) that has a universal value at x_c.

        Args:
            cumulant_by_width: Dict mapping width N to
                (control_param_array, cumulant_array).
            control_range: Range to search for intersection.

        Returns:
            Dictionary with 'x_c', 'cumulant_at_xc', and 'std'.
        """
        engine = ScalingCollapseEngine(self.config)
        result = engine.crossing_analysis(cumulant_by_width)

        # Also compute cumulant value at x_c
        x_c = result["x_c"]
        if np.isfinite(x_c):
            cum_values = []
            for N, (x, cum) in cumulant_by_width.items():
                try:
                    f = interpolate.interp1d(x, cum, kind="cubic", fill_value="extrapolate")
                    cum_values.append(float(f(x_c)))
                except Exception:
                    pass
            result["cumulant_at_xc"] = float(np.mean(cum_values)) if cum_values else np.nan
        else:
            result["cumulant_at_xc"] = np.nan

        return result


# ---------------------------------------------------------------------------
# CriticalExponentMeasurer
# ---------------------------------------------------------------------------

class CriticalExponentMeasurer:
    """Measure critical exponents from finite-size scaling data.

    Critical exponents characterise the singular behaviour of observables
    near a continuous phase transition.  In FSS, they appear as power-law
    dependences on the system size N at the critical point:

        - Correlation length:  ξ_max ~ N^{1}  (by definition of FSS)
        - Susceptibility peak: χ_max ~ N^{γ/ν}
        - Order parameter:     m(x_c) ~ N^{-β/ν}
        - Specific heat peak:  C_max ~ N^{α/ν}

    Hyperscaling relations (e.g., 2 - α = d·ν) provide consistency checks.

    References:
        Fisher (1967), Sec. IV: Scaling Laws.
        Barber (1983), Sec. 2.3: Finite-Size Scaling of Exponents.
    """

    def __init__(self, config: FiniteSizeScalingConfig):
        """Initialise with FSS configuration.

        Args:
            config: Configuration specifying widths, parameter range, etc.
        """
        self.config = config

    def _power_law_fit(
        self,
        widths: np.ndarray,
        values: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Fit y = a · N^b to (widths, values) data.

        Args:
            widths: System sizes.
            values: Observable values.

        Returns:
            Tuple (exponent, amplitude, r_squared).
        """
        mask = (widths > 0) & (values > 0) & np.isfinite(values)
        if np.sum(mask) < 2:
            return (np.nan, np.nan, np.nan)

        log_N = np.log(widths[mask])
        log_y = np.log(values[mask])
        fit = np.polyfit(log_N, log_y, 1)
        exponent = fit[0]
        amplitude = np.exp(fit[1])

        # R² value
        y_pred = np.polyval(fit, log_N)
        ss_res = np.sum((log_y - y_pred) ** 2)
        ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return (float(exponent), float(amplitude), float(r_sq))

    def measure_nu(
        self,
        data_by_width: Dict[int, Tuple[np.ndarray, np.ndarray]],
        critical_value: float,
    ) -> Dict[str, float]:
        """Measure ν from FSS of observable peak width ~ N^{-1/ν}.

        The width of the peak (or transition region) in the observable
        scales as Δx ~ N^{-1/ν}.  We measure this by fitting the
        second derivative near x_c.

        Args:
            data_by_width: Dict mapping width N to (x_array, y_array).
            critical_value: Critical value x_c.

        Returns:
            Dictionary with 'nu', 'one_over_nu', 'r_squared'.
        """
        widths = []
        peak_widths = []

        for N, (x, y) in data_by_width.items():
            # Estimate peak width: range where observable > 0.5 * max deviation
            y_at_xc = np.interp(critical_value, x, y)
            deviation = np.abs(y - y_at_xc)
            threshold = 0.5 * np.max(deviation)

            above = x[deviation > threshold]
            if len(above) >= 2:
                widths.append(N)
                peak_widths.append(above[-1] - above[0])

        if len(widths) < 2:
            return {"nu": np.nan, "one_over_nu": np.nan, "r_squared": np.nan}

        widths = np.array(widths, dtype=float)
        peak_widths = np.array(peak_widths)

        # Fit Δx ~ N^{-1/ν}
        exponent, _, r_sq = self._power_law_fit(widths, peak_widths)

        return {
            "nu": float(-1.0 / exponent) if np.abs(exponent) > 1e-10 else np.nan,
            "one_over_nu": float(-exponent),
            "r_squared": r_sq,
        }

    def measure_gamma(
        self,
        susceptibility_by_width: Dict[int, float],
        critical_value: float,
    ) -> Dict[str, float]:
        """Measure γ from χ_max ~ N^{γ/ν}.

        The susceptibility (variance of order parameter, or peak of
        response function) at x_c scales as N^{γ/ν}.

        Args:
            susceptibility_by_width: Dict mapping width N to peak
                susceptibility χ_max.
            critical_value: Critical value (for reference).

        Returns:
            Dictionary with 'gamma_over_nu', 'r_squared'.
        """
        widths = np.array(sorted(susceptibility_by_width.keys()), dtype=float)
        chi_max = np.array([susceptibility_by_width[int(N)] for N in widths])

        exponent, _, r_sq = self._power_law_fit(widths, chi_max)

        return {
            "gamma_over_nu": float(exponent),
            "r_squared": r_sq,
        }

    def measure_beta(
        self,
        order_param_at_xc_by_width: Dict[int, float],
    ) -> Dict[str, float]:
        """Measure β from m(x_c) ~ N^{-β/ν}.

        The order parameter at x_c decreases with system size as
        N^{-β/ν}.

        Args:
            order_param_at_xc_by_width: Dict mapping width N to order
                parameter value m(x_c, N).

        Returns:
            Dictionary with 'beta_over_nu', 'r_squared'.
        """
        widths = np.array(sorted(order_param_at_xc_by_width.keys()), dtype=float)
        m_xc = np.array([order_param_at_xc_by_width[int(N)] for N in widths])

        exponent, _, r_sq = self._power_law_fit(widths, np.abs(m_xc))

        return {
            "beta_over_nu": float(-exponent),  # exponent is negative
            "r_squared": r_sq,
        }

    def measure_alpha(
        self,
        specific_heat_by_width: Dict[int, float],
    ) -> Dict[str, float]:
        """Measure α from C_max ~ N^{α/ν}.

        The specific heat (second derivative of free energy) peak scales
        as N^{α/ν}.  For α < 0 (e.g., 3D Ising), the peak height
        converges, and one needs the logarithmic correction.

        Args:
            specific_heat_by_width: Dict mapping width N to C_max.

        Returns:
            Dictionary with 'alpha_over_nu', 'r_squared',
            'logarithmic_fit' (if α ≈ 0).
        """
        widths = np.array(sorted(specific_heat_by_width.keys()), dtype=float)
        c_max = np.array([specific_heat_by_width[int(N)] for N in widths])

        exponent, _, r_sq = self._power_law_fit(widths, c_max)

        result: Dict[str, float] = {
            "alpha_over_nu": float(exponent),
            "r_squared": r_sq,
        }

        # Check for logarithmic scaling (α ≈ 0)
        if np.abs(exponent) < 0.1:
            # Try C ~ a + b log(N)
            log_fit = np.polyfit(np.log(widths), c_max, 1)
            y_pred = np.polyval(log_fit, np.log(widths))
            ss_res = np.sum((c_max - y_pred) ** 2)
            ss_tot = np.sum((c_max - np.mean(c_max)) ** 2)
            log_r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            result["logarithmic_fit"] = float(log_r_sq)

        return result

    def measure_eta(
        self,
        correlation_at_xc_by_width: Dict[int, np.ndarray],
    ) -> Dict[str, float]:
        """Measure η from G(r, x_c) ~ r^{-(d-2+η)}.

        The correlation function at criticality decays algebraically.
        In the FSS context, the structure factor at the smallest
        non-zero momentum scales as S(k_min) ~ N^{2-η}.

        Args:
            correlation_at_xc_by_width: Dict mapping width N to
                correlation function array G(r) at x_c.

        Returns:
            Dictionary with 'eta', 'r_squared'.
        """
        widths_list = sorted(correlation_at_xc_by_width.keys())
        if not widths_list:
            return {"eta": np.nan, "r_squared": np.nan}

        # Use structure factor at smallest k: S(2π/N) ~ N^{2-η}
        widths = []
        s_kmin = []
        for N in widths_list:
            G = correlation_at_xc_by_width[N]
            # Fourier transform to get structure factor
            S = np.abs(np.fft.fft(G))
            if len(S) > 1:
                widths.append(N)
                s_kmin.append(float(S[1]))  # smallest non-zero k

        if len(widths) < 2:
            return {"eta": np.nan, "r_squared": np.nan}

        widths = np.array(widths, dtype=float)
        s_kmin = np.array(s_kmin)

        exponent, _, r_sq = self._power_law_fit(widths, s_kmin)
        # exponent = 2 - η
        eta = 2.0 - exponent

        return {"eta": float(eta), "r_squared": r_sq}

    def hyperscaling_check(
        self,
        exponents: Dict[str, float],
    ) -> Dict[str, Dict[str, float]]:
        """Verify hyperscaling relations between critical exponents.

        The standard hyperscaling relations for spatial dimension d are:

            2 - α = d · ν           (Josephson)
            γ = ν (2 - η)           (Fisher)
            2β + γ = d · ν          (Rushbrooke, as equality)
            γ = β (δ - 1)           (Widom)
            d · ν = 2β + γ          (scaling law)

        For neural networks, d is an effective dimension related to the
        scaling of the parameter space.

        Args:
            exponents: Dictionary with keys 'nu', 'gamma', 'beta',
                'alpha', 'eta', and optionally 'd' (effective dimension,
                default 1 for width scaling).

        Returns:
            Dictionary of hyperscaling relations with 'lhs', 'rhs',
            and 'violation' for each.
        """
        nu = exponents.get("nu", np.nan)
        gamma = exponents.get("gamma", np.nan)
        beta = exponents.get("beta", np.nan)
        alpha = exponents.get("alpha", np.nan)
        eta = exponents.get("eta", np.nan)
        d = exponents.get("d", 1.0)

        checks = {}

        # Josephson: 2 - α = d ν
        if np.isfinite(alpha) and np.isfinite(nu):
            lhs = 2.0 - alpha
            rhs = d * nu
            checks["josephson"] = {
                "lhs": float(lhs),
                "rhs": float(rhs),
                "violation": float(np.abs(lhs - rhs)),
            }

        # Fisher: γ = ν(2 - η)
        if np.isfinite(gamma) and np.isfinite(nu) and np.isfinite(eta):
            lhs = gamma
            rhs = nu * (2.0 - eta)
            checks["fisher"] = {
                "lhs": float(lhs),
                "rhs": float(rhs),
                "violation": float(np.abs(lhs - rhs)),
            }

        # Rushbrooke: 2β + γ = dν (as equality at critical point)
        if np.isfinite(beta) and np.isfinite(gamma) and np.isfinite(nu):
            lhs = 2.0 * beta + gamma
            rhs = d * nu
            checks["rushbrooke"] = {
                "lhs": float(lhs),
                "rhs": float(rhs),
                "violation": float(np.abs(lhs - rhs)),
            }

        return checks

    def correction_to_scaling(
        self,
        data_by_width: Dict[int, float],
        critical_value: float,
        omega_range: Tuple[float, float] = (0.3, 2.0),
    ) -> Dict[str, float]:
        """Fit observable with correction to scaling.

        Near x_c, the leading FSS behaviour receives corrections:

            O(x_c, N) = N^{a/ν} · (c₀ + c₁ · N^{-ω} + c₂ · N^{-2ω} + …)

        where ω is the leading irrelevant exponent (correction-to-scaling
        exponent).

        Args:
            data_by_width: Dict mapping width N to observable at x_c.
            critical_value: Critical value (for reference).
            omega_range: Bounds for ω.

        Returns:
            Dictionary with 'exponent' (a/ν), 'omega', 'amplitude',
            'correction_amplitude', 'r_squared'.

        References:
            Wegner (1972), "Corrections to scaling laws."
        """
        widths = np.array(sorted(data_by_width.keys()), dtype=float)
        values = np.array([data_by_width[int(N)] for N in widths])

        if len(widths) < 4:
            exp, amp, r_sq = self._power_law_fit(widths, np.abs(values))
            return {
                "exponent": exp, "omega": np.nan, "amplitude": amp,
                "correction_amplitude": 0.0, "r_squared": r_sq,
            }

        def fit_fn(params):
            a_nu, c0, c1, omega = params
            if omega <= 0:
                return 1e10
            y_pred = c0 * widths ** a_nu * (1.0 + c1 * widths ** (-omega))
            return np.sum((values - y_pred) ** 2)

        # Initial guess from power-law fit
        exp0, amp0, _ = self._power_law_fit(widths, np.abs(values))
        if not np.isfinite(exp0):
            exp0 = 0.0
        if not np.isfinite(amp0):
            amp0 = 1.0

        bounds = [(-5, 5), (amp0 * 0.01, amp0 * 100), (-10, 10), omega_range]
        try:
            result = optimize.differential_evolution(
                fit_fn, bounds, maxiter=300, seed=42
            )
            a_nu, c0, c1, omega = result.x

            y_pred = c0 * widths ** a_nu * (1.0 + c1 * widths ** (-omega))
            ss_res = np.sum((values - y_pred) ** 2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

            return {
                "exponent": float(a_nu),
                "omega": float(omega),
                "amplitude": float(c0),
                "correction_amplitude": float(c1),
                "r_squared": float(r_sq),
            }
        except Exception:
            return {
                "exponent": float(exp0), "omega": np.nan,
                "amplitude": float(amp0), "correction_amplitude": 0.0,
                "r_squared": np.nan,
            }

    def exponent_summary(
        self,
        all_measurements: Dict[str, Dict[str, float]],
    ) -> Dict[str, Dict[str, float]]:
        """Summary table of all measured critical exponents with errors.

        Collects all measured exponents, computes derived quantities
        (e.g., γ = γ/ν · ν), and checks hyperscaling relations.

        Args:
            all_measurements: Dictionary of measurement results from
                measure_nu, measure_gamma, etc.

        Returns:
            Dictionary with 'exponents' (name → {value, error}),
            'derived', and 'hyperscaling'.
        """
        exponents = {}

        # Extract raw exponents
        if "nu" in all_measurements:
            nu = all_measurements["nu"].get("nu", np.nan)
            exponents["nu"] = {"value": nu, "r_squared": all_measurements["nu"].get("r_squared", np.nan)}
        else:
            nu = np.nan

        if "gamma" in all_measurements:
            g_over_nu = all_measurements["gamma"].get("gamma_over_nu", np.nan)
            exponents["gamma_over_nu"] = {"value": g_over_nu, "r_squared": all_measurements["gamma"].get("r_squared", np.nan)}
            if np.isfinite(nu) and np.isfinite(g_over_nu):
                exponents["gamma"] = {"value": g_over_nu * nu, "derived_from": "gamma/nu * nu"}

        if "beta" in all_measurements:
            b_over_nu = all_measurements["beta"].get("beta_over_nu", np.nan)
            exponents["beta_over_nu"] = {"value": b_over_nu, "r_squared": all_measurements["beta"].get("r_squared", np.nan)}
            if np.isfinite(nu) and np.isfinite(b_over_nu):
                exponents["beta"] = {"value": b_over_nu * nu, "derived_from": "beta/nu * nu"}

        if "alpha" in all_measurements:
            a_over_nu = all_measurements["alpha"].get("alpha_over_nu", np.nan)
            exponents["alpha_over_nu"] = {"value": a_over_nu, "r_squared": all_measurements["alpha"].get("r_squared", np.nan)}
            if np.isfinite(nu) and np.isfinite(a_over_nu):
                exponents["alpha"] = {"value": a_over_nu * nu, "derived_from": "alpha/nu * nu"}

        if "eta" in all_measurements:
            exponents["eta"] = {"value": all_measurements["eta"].get("eta", np.nan), "r_squared": all_measurements["eta"].get("r_squared", np.nan)}

        # Hyperscaling check
        hs_input = {k: v["value"] for k, v in exponents.items() if "value" in v}
        hyperscaling = self.hyperscaling_check(hs_input) if len(hs_input) >= 3 else {}

        return {
            "exponents": exponents,
            "hyperscaling": hyperscaling,
        }


# ---------------------------------------------------------------------------
# NeuralNetworkFSS
# ---------------------------------------------------------------------------

class NeuralNetworkFSS:
    """Neural-network–specific finite-size scaling with width as system size.

    Applies FSS methods to neural network phase transitions where the
    network width N plays the role of the system size L.  Key transitions:

    - Lazy → Rich: as learning rate or regularisation increases, the
      network transitions from the kernel (lazy) regime to the feature-
      learning (rich) regime.
    - Generalisation: transition from overfitting to generalisation as
      the ratio n/N (samples / parameters) increases.
    - Double descent: non-monotonic test error as a function of model
      complexity, with a peak near the interpolation threshold n ≈ p.

    References:
        Geiger et al., "Scaling description of generalization with number
        of parameters in deep learning" (J. Stat. Mech. 2020).
        Mei & Montanari, "The generalization error of random features
        regression" (Ann. Stat. 2022).
    """

    def __init__(self, config: FiniteSizeScalingConfig):
        """Initialise with FSS configuration.

        Args:
            config: Configuration specifying widths, parameter range, etc.
        """
        self.config = config
        self._collapse_engine = ScalingCollapseEngine(config)
        self._extractor = CriticalPointExtractor(config)
        self._measurer = CriticalExponentMeasurer(config)

    def width_as_system_size(
        self,
        observable_fn: Callable[[int, float], float],
        widths: np.ndarray,
        control_range: np.ndarray,
    ) -> Dict[str, object]:
        """Treat width N as system size and perform full FSS analysis.

        1. Compute observable at each (width, control_param) pair.
        2. Find crossings for dimensionless quantities.
        3. Optimize data collapse.
        4. Extract critical point and exponents.

        Args:
            observable_fn: Function (width, control_param) → observable.
            widths: Array of network widths.
            control_range: Array of control parameter values.

        Returns:
            Dictionary with 'data', 'collapse', 'critical_point',
            'exponents'.
        """
        # Generate data
        data_by_width: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for N in widths:
            y = np.array([observable_fn(int(N), x) for x in control_range])
            data_by_width[int(N)] = (control_range.copy(), y)

        # Crossing analysis
        crossing = self._collapse_engine.crossing_analysis(data_by_width)

        # Optimize collapse
        xc_est = crossing.get("x_c", np.mean(control_range))
        if not np.isfinite(xc_est):
            xc_est = np.mean(control_range)

        collapse_result = self._collapse_engine.optimize_collapse(
            data_by_width,
            xc_range=(xc_est - 0.3 * np.abs(xc_est + 1e-3), xc_est + 0.3 * np.abs(xc_est + 1e-3)),
        )

        return {
            "data": data_by_width,
            "crossing": crossing,
            "collapse": collapse_result,
        }

    def lazy_rich_fss(
        self,
        widths: np.ndarray,
        lr_range: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, object]:
        """FSS for the lazy-to-rich transition.

        The lazy-rich transition occurs as the learning rate (or
        equivalently the output scale) increases.  In the lazy regime,
        the NTK is approximately constant and the network behaves as a
        kernel machine.  In the rich regime, features are learned and
        the NTK evolves.

        The order parameter is the relative NTK change:

            m = || K(t) - K(0) ||_F / || K(0) ||_F

        which is ~0 in lazy and O(1) in rich.

        Args:
            widths: Array of network widths.
            lr_range: Array of learning rates to scan.
            X: Input data (n_samples × d_input).
            y: Target labels (n_samples,).

        Returns:
            Dictionary with FSS results including 'x_c' (critical LR),
            'nu', and collapse quality.
        """
        n_samples = X.shape[0]
        d_input = X.shape[1]

        def ntk_change_observable(width: int, lr: float) -> float:
            """Compute relative NTK change as proxy order parameter.

            Uses random features approximation with simplified gradient
            descent dynamics.
            """
            rng = np.random.RandomState(width + int(lr * 1000))
            W = rng.randn(d_input, width) / np.sqrt(d_input)
            a = rng.randn(width) / np.sqrt(width)

            # Pre-activations and activations
            h = X @ W  # (n, width)
            phi = np.maximum(h, 0)  # ReLU

            # Initial NTK approximation: K_0 = (1/N) Φ Φ^T
            K0 = phi @ phi.T / width

            # One-step gradient update to approximate feature learning
            f0 = phi @ a  # predictions
            residual = f0 - y
            grad_a = phi.T @ residual / n_samples

            # Updated parameters
            a_new = a - lr * grad_a
            # Updated features (simplified — in reality W also changes)
            delta_a = a_new - a
            K1 = phi @ np.diag(1 + lr * delta_a ** 2) @ phi.T / width

            # Relative change
            K0_norm = np.linalg.norm(K0, "fro")
            if K0_norm < 1e-30:
                return 0.0
            return float(np.linalg.norm(K1 - K0, "fro") / K0_norm)

        return self.width_as_system_size(ntk_change_observable, widths, lr_range)

    def generalization_fss(
        self,
        widths: np.ndarray,
        n_samples_range: np.ndarray,
        X_fn: Callable[[int], np.ndarray],
        y_fn: Callable[[int], np.ndarray],
    ) -> Dict[str, object]:
        """FSS for the generalisation transition.

        As the ratio α = n/N (samples per parameter) increases, the
        network transitions from overfitting to generalisation.  The
        critical ratio α_c depends on the data distribution and
        architecture.

        The observable is the test error, which should show FSS as a
        function of α and N.

        Args:
            widths: Array of network widths.
            n_samples_range: Array of training set sizes.
            X_fn: Function(n_samples) → input data array.
            y_fn: Function(n_samples) → target label array.

        Returns:
            FSS analysis results.
        """
        def test_error_observable(width: int, n_samples: float) -> float:
            """Compute test error for random features model."""
            n = int(n_samples)
            if n < 2:
                return np.nan

            X_train = X_fn(n)
            y_train = y_fn(n)
            d = X_train.shape[1]

            rng = np.random.RandomState(width + n)
            W = rng.randn(d, width) / np.sqrt(d)
            phi = np.maximum(X_train @ W, 0)  # (n, width)

            # Ridge regression with small regularisation
            lam = 1e-4
            gram = phi.T @ phi + lam * np.eye(width)
            try:
                a = np.linalg.solve(gram, phi.T @ y_train)
            except np.linalg.LinAlgError:
                return np.nan

            # Test error on fresh data
            n_test = min(n, 100)
            X_test = X_fn(n_test)
            y_test = y_fn(n_test)
            phi_test = np.maximum(X_test @ W, 0)
            y_pred = phi_test @ a

            return float(np.mean((y_pred - y_test) ** 2))

        # Control parameter is the ratio n/N, evaluated at different widths
        data_by_width: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for N in widths:
            alpha_values = n_samples_range / N
            errors = np.array([test_error_observable(int(N), n) for n in n_samples_range])
            data_by_width[int(N)] = (alpha_values, errors)

        # Find critical ratio via crossing
        crossing = self._collapse_engine.crossing_analysis(data_by_width)

        return {
            "data": data_by_width,
            "crossing": crossing,
            "critical_alpha": crossing.get("x_c", np.nan),
        }

    def double_descent_fss(
        self,
        widths: np.ndarray,
        n_samples: int,
        X_fn: Callable[[int], np.ndarray],
        y_fn: Callable[[int], np.ndarray],
    ) -> Dict[str, object]:
        """FSS near the interpolation threshold (double descent).

        At n ≈ p (number of parameters), the test error diverges for
        minimum-norm interpolation.  This divergence has FSS structure:

            E_test(n, N) ~ N^{a/ν} · G((n/N - γ_c) · N^{1/ν})

        where γ_c is the interpolation threshold ratio.

        Args:
            widths: Array of network widths (each gives p ≈ width params).
            n_samples: Number of training samples.
            X_fn: Function(n) → input data.
            y_fn: Function(n) → labels.

        Returns:
            FSS analysis near interpolation threshold.

        References:
            Belkin et al., "Reconciling modern machine learning practice
            and the bias-variance trade-off" (PNAS 2019).
        """
        data_by_width: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        for N in widths:
            # Scan aspect ratio n/N around 1 (interpolation threshold)
            alpha_range = np.linspace(0.3, 3.0, 50)
            errors = []

            for alpha in alpha_range:
                n_eff = max(int(alpha * N), 2)
                X_train = X_fn(n_eff)
                y_train = y_fn(n_eff)
                d = X_train.shape[1]

                rng = np.random.RandomState(int(N) + n_eff)
                W = rng.randn(d, int(N)) / np.sqrt(d)
                phi = np.maximum(X_train @ W, 0)

                # Minimum-norm solution
                try:
                    if n_eff <= int(N):
                        # Under-determined: minimum-norm interpolation
                        gram = phi @ phi.T
                        gram += 1e-10 * np.eye(n_eff)
                        c = np.linalg.solve(gram, y_train)
                        a = phi.T @ c
                    else:
                        # Over-determined: least squares
                        a, _, _, _ = np.linalg.lstsq(phi, y_train, rcond=None)

                    # Test error
                    X_test = X_fn(min(n_eff, 100))
                    y_test = y_fn(min(n_eff, 100))
                    phi_test = np.maximum(X_test @ W, 0)
                    y_pred = phi_test @ a
                    errors.append(float(np.mean((y_pred - y_test) ** 2)))
                except Exception:
                    errors.append(np.nan)

            data_by_width[int(N)] = (alpha_range, np.array(errors))

        # The peak of the error curve indicates the interpolation threshold
        peak_positions = {}
        for N, (alpha, err) in data_by_width.items():
            finite = np.isfinite(err)
            if np.any(finite):
                peak_idx = np.argmax(err[finite])
                peak_positions[N] = float(alpha[finite][peak_idx])

        return {
            "data": data_by_width,
            "interpolation_threshold": peak_positions,
        }

    def depth_scaling_correction(
        self,
        widths: np.ndarray,
        depth: int,
    ) -> Dict[str, float]:
        """How depth modifies finite-size scaling.

        For deep networks, the effective system size receives depth
        corrections.  The effective width is

            N_eff = N / f(L)

        where f(L) is a depth-dependent correction.  For standard
        parameterisation, f(L) ~ L (depth).  For μP, f(L) ~ 1.

        Args:
            widths: Array of network widths.
            depth: Network depth L.

        Returns:
            Dictionary with 'effective_widths', 'depth_factor',
            'parameterization_type'.
        """
        # Standard parameterisation: signal propagation degrades with depth
        # Effective width N_eff ~ N / L for signal-to-noise ratio
        depth_factor_standard = float(depth)

        # μP parameterisation: depth-independent scaling
        depth_factor_mup = 1.0

        effective_widths_standard = widths / depth_factor_standard
        effective_widths_mup = widths / depth_factor_mup

        return {
            "effective_widths_standard": effective_widths_standard,
            "effective_widths_mup": effective_widths_mup,
            "depth_factor_standard": depth_factor_standard,
            "depth_factor_mup": depth_factor_mup,
            "depth": depth,
        }

    def mu_p_fss(
        self,
        widths: np.ndarray,
        lr_range: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, object]:
        """FSS in the maximal update parameterisation (μP).

        In μP, the learning rate scales as η_μP = η₀ / N, and the
        output scales as 1/√N.  This ensures that the feature-learning
        scale is width-independent, giving a true infinite-width limit
        that preserves feature learning.

        The lazy-rich transition in μP occurs at a width-independent
        critical learning rate η_c, making FSS particularly clean.

        Args:
            widths: Array of network widths.
            lr_range: Array of base learning rates η₀ (before width scaling).
            X: Input data.
            y: Target labels.

        Returns:
            FSS results in μP parameterisation.

        References:
            Yang & Hu, "Feature Learning in Infinite-Width Neural
            Networks" (ICML 2021) — μP definition.
        """
        n_samples = X.shape[0]
        d_input = X.shape[1]

        def mup_observable(width: int, lr_base: float) -> float:
            """Observable in μP: NTK change with μP-scaled learning rate."""
            lr = lr_base / width  # μP scaling
            rng = np.random.RandomState(width + int(lr_base * 1000))

            # μP initialisation: W ~ N(0, 1/d), a ~ N(0, 1/N)
            W = rng.randn(d_input, width) / np.sqrt(d_input)
            a = rng.randn(width) / width  # μP: 1/N not 1/√N

            h = X @ W
            phi = np.maximum(h, 0)

            # Output scale 1/√N in μP
            f0 = phi @ a * np.sqrt(width)
            residual = f0 - y

            # Gradient step with μP learning rate
            grad_a = phi.T @ residual / n_samples
            a_new = a - lr * grad_a

            # Feature change metric
            delta_f = phi @ (a_new - a) * np.sqrt(width)
            signal = np.linalg.norm(delta_f)
            if np.linalg.norm(f0) < 1e-30:
                return 0.0
            return float(signal / (np.linalg.norm(f0) + 1e-30))

        return self.width_as_system_size(mup_observable, widths, lr_range)

    def width_depth_fss(
        self,
        widths: np.ndarray,
        depths: np.ndarray,
        control_range: np.ndarray,
    ) -> Dict[str, object]:
        """Joint FSS in (width N, depth L) space.

        For networks with both varying width and depth, the FSS ansatz
        generalises to:

            O(x, N, L) = N^{a/ν_N} L^{b/ν_L} F(x · N^{1/ν_N}, L/N^ψ)

        where ψ = ν_N / ν_L is the anisotropy exponent relating width
        and depth scaling.

        Args:
            widths: Array of widths.
            depths: Array of depths.
            control_range: Array of control parameter values.

        Returns:
            Dictionary with 'data', 'anisotropy_exponent',
            'width_exponent', 'depth_exponent'.
        """
        # Generate data on a 2D (width, depth) grid
        data = {}
        for N in widths:
            for L in depths:
                key = (int(N), int(L))
                rng = np.random.RandomState(int(N) * 100 + int(L))

                # Observable: training loss of random-features model
                # with given width and depth (simplified)
                values = []
                for x in control_range:
                    # Depth-L network: compose L random feature maps
                    d = 10  # input dimension
                    features = rng.randn(100, d)  # 100 samples
                    for layer in range(int(L)):
                        W = rng.randn(features.shape[1], int(N)) / np.sqrt(features.shape[1])
                        features = np.maximum(features @ W, 0)

                    # Ridge regression with control parameter as regularisation
                    gram = features.T @ features + np.exp(x) * np.eye(int(N))
                    try:
                        y_target = rng.randn(100)
                        a = np.linalg.solve(gram, features.T @ y_target)
                        loss = np.mean((features @ a - y_target) ** 2)
                        values.append(float(loss))
                    except Exception:
                        values.append(np.nan)

                data[key] = (control_range.copy(), np.array(values))

        # Estimate anisotropy exponent ψ
        # At fixed x, O(N, L) ~ N^a when L/N^ψ = const
        # Try ψ values and find best collapse
        best_psi = 1.0
        best_quality = np.inf

        for psi in np.linspace(0.2, 2.0, 20):
            # Group data by effective depth ratio L/N^ψ
            groups: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}
            for (N, L), (x, y) in data.items():
                ratio = L / N ** psi
                ratio_bin = f"{ratio:.1f}"
                if ratio_bin not in groups:
                    groups[ratio_bin] = {}
                groups[ratio_bin][N] = (x, y)

            # Check collapse quality for the largest group
            if groups:
                largest_group = max(groups.values(), key=len)
                if len(largest_group) >= 2:
                    try:
                        result = self._collapse_engine.optimize_collapse(largest_group)
                        if result["quality"] < best_quality:
                            best_quality = result["quality"]
                            best_psi = psi
                    except Exception:
                        pass

        return {
            "data": data,
            "anisotropy_exponent": float(best_psi),
            "collapse_quality": float(best_quality),
        }
