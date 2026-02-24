"""
Finite-size scaling analysis for neural networks.

Implements finite-size scaling ansatz, data collapse optimization,
crossing analysis, Binder cumulant, correction to scaling,
and bootstrap error estimation.
"""

import numpy as np
from scipy.optimize import minimize, least_squares, brentq
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.stats import bootstrap as scipy_bootstrap
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Callable
import warnings


@dataclass
class ScalingReport:
    """Report from finite-size scaling analysis."""
    critical_point: float = 0.0
    exponents: Dict[str, float] = field(default_factory=dict)
    quality_of_collapse: float = 0.0
    scaling_function: Dict[str, Any] = field(default_factory=dict)
    crossing_points: List[Dict[str, float]] = field(default_factory=list)
    binder_cumulant_crossing: float = 0.0
    corrections_to_scaling: Dict[str, float] = field(default_factory=dict)
    bootstrap_errors: Dict[str, float] = field(default_factory=dict)
    data_collapse_params: Dict[str, float] = field(default_factory=dict)


class DataCollapseOptimizer:
    """Optimize data collapse for finite-size scaling."""

    def __init__(self, collapse_method: str = "residual"):
        self.collapse_method = collapse_method

    def collapse_quality(self, params: np.ndarray,
                          sizes: List[int],
                          temperatures: List[np.ndarray],
                          observables: List[np.ndarray],
                          scaling_form: str = "standard") -> float:
        """Compute quality of data collapse.

        Standard ansatz: O(T, L) = L^(a/nu) * f((T - Tc) * L^(1/nu))
        """
        Tc = params[0]
        nu = max(0.01, params[1])
        a_over_nu = params[2] if len(params) > 2 else 0.0

        all_x_scaled = []
        all_y_scaled = []
        all_size_labels = []

        for i, (L, T_arr, O_arr) in enumerate(zip(sizes, temperatures, observables)):
            x_scaled = (T_arr - Tc) * L ** (1.0 / nu)
            if scaling_form == "standard":
                y_scaled = O_arr / (L ** a_over_nu)
            elif scaling_form == "logarithmic":
                y_scaled = O_arr / np.log(L + 1)
            else:
                y_scaled = O_arr

            all_x_scaled.extend(x_scaled.tolist())
            all_y_scaled.extend(y_scaled.tolist())
            all_size_labels.extend([L] * len(x_scaled))

        all_x = np.array(all_x_scaled)
        all_y = np.array(all_y_scaled)
        sort_idx = np.argsort(all_x)
        all_x = all_x[sort_idx]
        all_y = all_y[sort_idx]

        if len(all_x) < 4:
            return 1e10

        if self.collapse_method == "residual":
            return self._residual_quality(all_x, all_y)
        elif self.collapse_method == "interpolation":
            return self._interpolation_quality(all_x, all_y, sizes, all_size_labels, sort_idx)
        else:
            return self._residual_quality(all_x, all_y)

    def _residual_quality(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute collapse quality via residuals between neighboring points."""
        residual = 0.0
        count = 0
        for i in range(1, len(x)):
            dx = abs(x[i] - x[i - 1])
            if dx > 1e-10:
                dy = abs(y[i] - y[i - 1])
                residual += (dy / (dx + 1e-10)) ** 2
                count += 1
        return residual / (count + 1) if count > 0 else 1e10

    def _interpolation_quality(self, x: np.ndarray, y: np.ndarray,
                                sizes: List[int], size_labels: List[int],
                                sort_idx: np.ndarray) -> float:
        """Compute collapse quality via comparison to master curve."""
        try:
            unique_x = np.unique(x)
            if len(unique_x) < 4:
                return 1e10

            y_smooth = np.zeros(len(unique_x))
            for i, ux in enumerate(unique_x):
                mask = np.abs(x - ux) < 1e-10
                if np.any(mask):
                    y_smooth[i] = np.mean(y[mask])
                else:
                    y_smooth[i] = np.interp(ux, x, y)

            master = interp1d(unique_x, y_smooth, kind='linear',
                             fill_value='extrapolate', bounds_error=False)

            residual = np.mean((y - master(x)) ** 2)
            return float(residual)
        except Exception:
            return self._residual_quality(x, y)

    def optimize(self, sizes: List[int],
                 temperatures: List[np.ndarray],
                 observables: List[np.ndarray],
                 Tc_range: Tuple[float, float] = (0.5, 3.0),
                 nu_range: Tuple[float, float] = (0.3, 3.0),
                 scaling_form: str = "standard") -> Dict[str, float]:
        """Find optimal data collapse parameters."""
        def objective(params):
            return self.collapse_quality(params, sizes, temperatures,
                                          observables, scaling_form)

        best_result = None
        best_obj = float("inf")

        for Tc_init in np.linspace(Tc_range[0], Tc_range[1], 5):
            for nu_init in np.linspace(nu_range[0], nu_range[1], 5):
                for a_init in [-1.0, 0.0, 1.0]:
                    try:
                        result = minimize(
                            objective,
                            x0=[Tc_init, nu_init, a_init],
                            bounds=[Tc_range, nu_range, (-5.0, 5.0)],
                            method="L-BFGS-B"
                        )
                        if result.fun < best_obj:
                            best_obj = result.fun
                            best_result = result
                    except Exception:
                        continue

        if best_result is None:
            return {"Tc": np.mean(Tc_range), "nu": 1.0, "a_over_nu": 0.0,
                    "quality": 0.0, "converged": False}

        quality = 1.0 / (1.0 + best_obj)
        return {
            "Tc": float(best_result.x[0]),
            "nu": float(best_result.x[1]),
            "a_over_nu": float(best_result.x[2]),
            "quality": float(quality),
            "converged": bool(best_result.success),
            "residual": float(best_obj),
        }


class CrossingAnalyzer:
    """Find crossing points where observable values coincide for different sizes."""

    def __init__(self):
        pass

    def find_crossing(self, T1: np.ndarray, O1: np.ndarray,
                      T2: np.ndarray, O2: np.ndarray) -> Optional[Dict[str, float]]:
        """Find crossing point of two curves."""
        T_min = max(T1.min(), T2.min())
        T_max = min(T1.max(), T2.max())
        if T_min >= T_max:
            return None

        T_common = np.linspace(T_min, T_max, 500)

        try:
            interp1 = interp1d(T1, O1, kind='linear', fill_value='extrapolate')
            interp2 = interp1d(T2, O2, kind='linear', fill_value='extrapolate')
        except ValueError:
            return None

        diff = interp1(T_common) - interp2(T_common)
        sign_changes = np.where(np.diff(np.sign(diff)))[0]

        if len(sign_changes) == 0:
            return None

        best_idx = sign_changes[0]
        try:
            Tc = brentq(lambda t: float(interp1(t) - interp2(t)),
                        T_common[best_idx], T_common[best_idx + 1])
            Oc = float(interp1(Tc))
            return {"Tc": float(Tc), "Oc": Oc}
        except (ValueError, RuntimeError):
            return {"Tc": float(T_common[best_idx]),
                    "Oc": float(interp1(T_common[best_idx]))}

    def find_all_crossings(self, sizes: List[int],
                            temperatures: List[np.ndarray],
                            observables: List[np.ndarray]) -> List[Dict[str, float]]:
        """Find all pairwise crossings."""
        crossings = []
        n = len(sizes)
        for i in range(n):
            for j in range(i + 1, n):
                crossing = self.find_crossing(
                    temperatures[i], observables[i],
                    temperatures[j], observables[j]
                )
                if crossing is not None:
                    crossing["L1"] = sizes[i]
                    crossing["L2"] = sizes[j]
                    crossings.append(crossing)
        return crossings

    def estimate_Tc_from_crossings(self, crossings: List[Dict[str, float]]) -> Dict[str, float]:
        """Estimate critical point from crossing analysis."""
        if not crossings:
            return {"Tc": 0.0, "Tc_error": float("inf"), "n_crossings": 0}

        Tc_values = [c["Tc"] for c in crossings]
        weights = [1.0 / (abs(c["L2"] - c["L1"]) + 1) for c in crossings]
        weights = np.array(weights)
        weights /= np.sum(weights)

        Tc = float(np.average(Tc_values, weights=weights))
        Tc_error = float(np.sqrt(np.average((np.array(Tc_values) - Tc) ** 2, weights=weights)))

        return {
            "Tc": Tc,
            "Tc_error": Tc_error,
            "n_crossings": len(crossings),
            "Tc_values": Tc_values,
        }


class BinderCumulant:
    """Binder cumulant analysis for phase transition detection."""

    def __init__(self):
        pass

    def compute(self, samples: np.ndarray) -> float:
        """Compute Binder cumulant U = 1 - <m^4> / (3 <m^2>^2)."""
        m2 = np.mean(samples ** 2)
        m4 = np.mean(samples ** 4)
        if m2 < 1e-12:
            return 0.0
        return float(1.0 - m4 / (3.0 * m2 ** 2))

    def compute_from_observable(self, observable_values: np.ndarray) -> float:
        """Compute Binder cumulant from observable samples."""
        return self.compute(observable_values)

    def compute_curve(self, temperatures: np.ndarray,
                      sample_fn: Callable,
                      n_samples_per_T: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Binder cumulant as function of temperature."""
        binder_values = np.zeros(len(temperatures))
        for i, T in enumerate(temperatures):
            samples = sample_fn(T, n_samples_per_T)
            binder_values[i] = self.compute(samples)
        return temperatures, binder_values

    def find_crossing_from_binder(
        self, sizes: List[int],
        temperatures: List[np.ndarray],
        binder_values: List[np.ndarray]
    ) -> Dict[str, float]:
        """Find Binder cumulant crossing to determine Tc."""
        analyzer = CrossingAnalyzer()
        crossings = analyzer.find_all_crossings(sizes, temperatures, binder_values)
        result = analyzer.estimate_Tc_from_crossings(crossings)

        result["binder_at_Tc"] = float(np.mean([c["Oc"] for c in crossings])) \
            if crossings else 0.0
        return result

    def generate_synthetic_binder(self, sizes: List[int], Tc: float,
                                   nu: float, T_range: Tuple[float, float],
                                   n_T: int = 50) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Generate synthetic Binder cumulant data for testing."""
        temperatures_list = []
        binder_list = []

        for L in sizes:
            T_vals = np.linspace(T_range[0], T_range[1], n_T)
            x = (T_vals - Tc) * L ** (1.0 / nu)
            binder = 2.0 / 3.0 * (1.0 - np.tanh(x / 2.0))
            noise = np.random.randn(n_T) * 0.01 / np.sqrt(L)
            binder += noise
            temperatures_list.append(T_vals)
            binder_list.append(binder)

        return temperatures_list, binder_list


class CorrectionToScaling:
    """Include correction-to-scaling terms in finite-size scaling."""

    def __init__(self):
        pass

    def scaling_with_correction(self, T: float, L: int, params: Dict[str, float]) -> float:
        """Compute observable with leading correction to scaling.

        O(T, L) = L^(a/nu) * [f_0(x) + L^(-omega) * f_1(x)]
        where x = (T - Tc) * L^(1/nu)
        """
        Tc = params.get("Tc", 1.0)
        nu = params.get("nu", 1.0)
        a_over_nu = params.get("a_over_nu", 0.0)
        omega = params.get("omega", 1.0)
        c0 = params.get("c0", 1.0)
        c1 = params.get("c1", 0.0)

        x = (T - Tc) * L ** (1.0 / nu)
        f0 = c0 * np.exp(-x ** 2 / 2)
        f1 = c1 * x * np.exp(-x ** 2 / 2)

        return L ** a_over_nu * (f0 + L ** (-omega) * f1)

    def fit_with_correction(self, sizes: List[int],
                             temperatures: List[np.ndarray],
                             observables: List[np.ndarray],
                             Tc_range: Tuple[float, float] = (0.5, 3.0),
                             nu_range: Tuple[float, float] = (0.3, 3.0)
                             ) -> Dict[str, float]:
        """Fit scaling with correction to scaling term."""
        def objective(params):
            Tc, nu, a_over_nu, omega, c0, c1 = params
            param_dict = {"Tc": Tc, "nu": nu, "a_over_nu": a_over_nu,
                          "omega": omega, "c0": c0, "c1": c1}
            residual = 0.0
            count = 0
            for L, T_arr, O_arr in zip(sizes, temperatures, observables):
                for T, O in zip(T_arr, O_arr):
                    predicted = self.scaling_with_correction(T, L, param_dict)
                    residual += (predicted - O) ** 2
                    count += 1
            return residual / (count + 1)

        best_result = None
        best_obj = float("inf")

        for Tc_init in np.linspace(Tc_range[0], Tc_range[1], 3):
            for nu_init in [0.5, 1.0, 2.0]:
                try:
                    result = minimize(
                        objective,
                        x0=[Tc_init, nu_init, 0.0, 1.0, 1.0, 0.0],
                        bounds=[Tc_range, nu_range, (-3, 3), (0.1, 5), (-10, 10), (-10, 10)],
                        method="L-BFGS-B"
                    )
                    if result.fun < best_obj:
                        best_obj = result.fun
                        best_result = result
                except Exception:
                    continue

        if best_result is None:
            return {"Tc": np.mean(Tc_range), "nu": 1.0, "omega": 1.0,
                    "a_over_nu": 0.0, "quality": 0.0}

        return {
            "Tc": float(best_result.x[0]),
            "nu": float(best_result.x[1]),
            "a_over_nu": float(best_result.x[2]),
            "omega": float(best_result.x[3]),
            "c0": float(best_result.x[4]),
            "c1": float(best_result.x[5]),
            "quality": float(1.0 / (1.0 + best_obj)),
        }

    def estimate_omega(self, sizes: List[int], Tc_values: List[float]) -> float:
        """Estimate correction exponent omega from Tc drift with size."""
        if len(sizes) < 3 or len(Tc_values) < 3:
            return 1.0

        log_L = np.log(np.array(sizes, dtype=float))
        Tc_arr = np.array(Tc_values)
        Tc_inf = Tc_arr[-1]
        dTc = np.abs(Tc_arr - Tc_inf)
        mask = dTc > 1e-8
        if np.sum(mask) < 2:
            return 1.0

        log_dTc = np.log(dTc[mask] + 1e-30)
        coeffs = np.polyfit(log_L[mask], log_dTc, 1)
        return float(max(0.1, -coeffs[0]))


class BootstrapErrorEstimator:
    """Bootstrap error estimation for scaling exponents."""

    def __init__(self, n_bootstrap: int = 200):
        self.n_bootstrap = n_bootstrap

    def estimate_errors(self, sizes: List[int],
                        temperatures: List[np.ndarray],
                        observables: List[np.ndarray],
                        Tc_range: Tuple[float, float] = (0.5, 3.0),
                        nu_range: Tuple[float, float] = (0.3, 3.0)
                        ) -> Dict[str, float]:
        """Estimate bootstrap errors for scaling exponents."""
        optimizer = DataCollapseOptimizer()
        Tc_samples = []
        nu_samples = []

        for _ in range(self.n_bootstrap):
            boot_temps = []
            boot_obs = []
            for T_arr, O_arr in zip(temperatures, observables):
                n = len(T_arr)
                indices = np.random.choice(n, size=n, replace=True)
                boot_temps.append(T_arr[indices])
                boot_obs.append(O_arr[indices])

            try:
                result = optimizer.optimize(sizes, boot_temps, boot_obs,
                                             Tc_range, nu_range)
                if result["converged"]:
                    Tc_samples.append(result["Tc"])
                    nu_samples.append(result["nu"])
            except Exception:
                continue

        if len(Tc_samples) < 10:
            return {"Tc_error": float("inf"), "nu_error": float("inf"),
                    "n_successful": len(Tc_samples)}

        Tc_samples = np.array(Tc_samples)
        nu_samples = np.array(nu_samples)

        return {
            "Tc_mean": float(np.mean(Tc_samples)),
            "Tc_error": float(np.std(Tc_samples)),
            "Tc_95ci": [float(np.percentile(Tc_samples, 2.5)),
                        float(np.percentile(Tc_samples, 97.5))],
            "nu_mean": float(np.mean(nu_samples)),
            "nu_error": float(np.std(nu_samples)),
            "nu_95ci": [float(np.percentile(nu_samples, 2.5)),
                        float(np.percentile(nu_samples, 97.5))],
            "n_successful": len(Tc_samples),
        }


class ScalingFunctionReconstructor:
    """Reconstruct the universal scaling function from data."""

    def __init__(self, n_bins: int = 50):
        self.n_bins = n_bins

    def reconstruct(self, sizes: List[int],
                    temperatures: List[np.ndarray],
                    observables: List[np.ndarray],
                    Tc: float, nu: float,
                    a_over_nu: float = 0.0) -> Dict[str, Any]:
        """Reconstruct scaling function from collapsed data."""
        all_x = []
        all_y = []

        for L, T_arr, O_arr in zip(sizes, temperatures, observables):
            x_scaled = (T_arr - Tc) * L ** (1.0 / nu)
            y_scaled = O_arr / (L ** a_over_nu) if a_over_nu != 0 else O_arr
            all_x.extend(x_scaled.tolist())
            all_y.extend(y_scaled.tolist())

        all_x = np.array(all_x)
        all_y = np.array(all_y)
        sort_idx = np.argsort(all_x)
        all_x = all_x[sort_idx]
        all_y = all_y[sort_idx]

        x_bins = np.linspace(all_x.min(), all_x.max(), self.n_bins + 1)
        x_centers = (x_bins[:-1] + x_bins[1:]) / 2
        y_binned = np.zeros(self.n_bins)
        y_errors = np.zeros(self.n_bins)

        for i in range(self.n_bins):
            mask = (all_x >= x_bins[i]) & (all_x < x_bins[i + 1])
            if np.any(mask):
                y_binned[i] = np.mean(all_y[mask])
                y_errors[i] = np.std(all_y[mask]) / np.sqrt(np.sum(mask))
            else:
                y_binned[i] = np.nan
                y_errors[i] = np.nan

        valid = ~np.isnan(y_binned)
        return {
            "x_scaled": x_centers[valid].tolist(),
            "y_scaled": y_binned[valid].tolist(),
            "y_errors": y_errors[valid].tolist(),
            "n_points": int(np.sum(valid)),
            "x_range": [float(all_x.min()), float(all_x.max())],
        }


def generate_synthetic_fss_data(
    Tc: float, nu: float, a_over_nu: float,
    sizes: List[int], T_range: Tuple[float, float],
    n_T: int = 30, noise_level: float = 0.02
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Generate synthetic finite-size scaling data for testing."""
    temperatures_list = []
    observables_list = []

    for L in sizes:
        T_vals = np.linspace(T_range[0], T_range[1], n_T)
        x = (T_vals - Tc) * L ** (1.0 / nu)
        f_x = np.exp(-x ** 2 / 2) * (1 + 0.2 * x)
        observable = L ** a_over_nu * f_x
        noise = np.random.randn(n_T) * noise_level * np.abs(observable).mean()
        observable += noise

        temperatures_list.append(T_vals)
        observables_list.append(observable)

    return temperatures_list, observables_list


class FiniteSizeScaler:
    """Main finite-size scaling analysis class."""

    def __init__(self, n_bootstrap: int = 100):
        self.optimizer = DataCollapseOptimizer()
        self.crossing = CrossingAnalyzer()
        self.binder = BinderCumulant()
        self.correction = CorrectionToScaling()
        self.bootstrap = BootstrapErrorEstimator(n_bootstrap)
        self.reconstructor = ScalingFunctionReconstructor()

    def analyze(self, observable: Dict[str, Any],
                sizes: List[int]) -> ScalingReport:
        """Full finite-size scaling analysis.

        observable should contain:
        - temperatures: list of np.ndarray (one per size)
        - values: list of np.ndarray (one per size)
        Optional:
        - Tc_range: (float, float) for critical point search
        - nu_range: (float, float) for exponent search
        """
        report = ScalingReport()

        temperatures = observable.get("temperatures", [])
        values = observable.get("values", [])
        Tc_range = observable.get("Tc_range", (0.5, 3.0))
        nu_range = observable.get("nu_range", (0.3, 3.0))

        if not temperatures or not values or len(temperatures) != len(sizes):
            warnings.warn("Invalid observable data format")
            return report

        collapse_result = self.optimizer.optimize(
            sizes, temperatures, values, Tc_range, nu_range
        )
        report.critical_point = collapse_result["Tc"]
        report.exponents = {
            "nu": collapse_result["nu"],
            "a_over_nu": collapse_result["a_over_nu"],
        }
        report.quality_of_collapse = collapse_result["quality"]
        report.data_collapse_params = collapse_result

        crossings = self.crossing.find_all_crossings(sizes, temperatures, values)
        report.crossing_points = crossings
        if crossings:
            crossing_result = self.crossing.estimate_Tc_from_crossings(crossings)
            report.critical_point = crossing_result["Tc"]

        binder_crossing = self.binder.find_crossing_from_binder(
            sizes, temperatures, values
        )
        report.binder_cumulant_crossing = binder_crossing.get("Tc", report.critical_point)

        correction_result = self.correction.fit_with_correction(
            sizes, temperatures, values, Tc_range, nu_range
        )
        report.corrections_to_scaling = correction_result

        bootstrap_result = self.bootstrap.estimate_errors(
            sizes, temperatures, values, Tc_range, nu_range
        )
        report.bootstrap_errors = bootstrap_result

        if collapse_result["quality"] > 0.1:
            sf = self.reconstructor.reconstruct(
                sizes, temperatures, values,
                collapse_result["Tc"], collapse_result["nu"],
                collapse_result["a_over_nu"]
            )
            report.scaling_function = sf

        return report

    def quick_analyze(self, sizes: List[int],
                      temperatures: List[np.ndarray],
                      observables: List[np.ndarray],
                      Tc_range: Tuple[float, float] = (0.5, 3.0)
                      ) -> Dict[str, float]:
        """Quick analysis without bootstrap."""
        collapse_result = self.optimizer.optimize(
            sizes, temperatures, observables, Tc_range
        )
        crossings = self.crossing.find_all_crossings(sizes, temperatures, observables)

        result = dict(collapse_result)
        if crossings:
            crossing_Tc = self.crossing.estimate_Tc_from_crossings(crossings)
            result["crossing_Tc"] = crossing_Tc["Tc"]
            result["crossing_Tc_error"] = crossing_Tc["Tc_error"]

        return result

    def compare_scaling_ansatze(self, sizes: List[int],
                                 temperatures: List[np.ndarray],
                                 observables: List[np.ndarray]) -> Dict[str, Any]:
        """Compare different scaling forms."""
        results = {}

        for form in ["standard", "logarithmic"]:
            opt = DataCollapseOptimizer()
            result = opt.optimize(sizes, temperatures, observables,
                                   scaling_form=form)
            results[form] = result

        best_form = max(results, key=lambda k: results[k]["quality"])
        return {
            "results": results,
            "best_form": best_form,
            "best_quality": results[best_form]["quality"],
        }
