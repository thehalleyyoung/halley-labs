"""
Aging and memory effects in neural network training dynamics.

Implements two-time correlation functions, fluctuation-dissipation theorem
violation analysis, and connections between aging phenomena and grokking.
Inspired by spin glass physics and out-of-equilibrium statistical mechanics.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional, Callable, Dict, Any
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import interp1d


@dataclass
class AgingConfig:
    """Configuration for aging and memory analysis.

    Attributes:
        max_time: Maximum observation time (training steps).
        n_waiting_times: Number of waiting times t_w to probe.
        n_observation_times: Number of observation times t to probe per t_w.
        temperature: Effective temperature (learning rate or noise scale).
    """
    max_time: int = 100000
    n_waiting_times: int = 20
    n_observation_times: int = 50
    temperature: float = 0.01


class TwoTimeCorrelation:
    """Two-time correlation and response functions for aging analysis.

    In aging systems, two-time quantities depend on both times separately,
    not just on their difference. The correlation C(t_w + t, t_w) and
    response R(t_w + t, t_w) encode the full non-equilibrium dynamics.

    Key signature of aging: C(t_w + t, t_w) ≠ C(t) — the system never
    equilibrates and its properties depend on its age t_w.

    Attributes:
        config: AgingConfig with time and temperature parameters.
    """

    def __init__(self, config: AgingConfig):
        """Initialize two-time correlation analyzer.

        Args:
            config: Configuration for aging analysis.
        """
        self.config = config

    def compute_correlation(
        self, trajectory: np.ndarray, t_w: int, t: int
    ) -> float:
        """Compute the two-time correlation C(t_w + t, t_w).

        C(t_w + t, t_w) = <φ(t_w + t) · φ(t_w)> / N

        where φ is the state vector (e.g. network weights) and N is
        the dimension for normalization.

        Args:
            trajectory: Array of shape (T, N) with state vectors at each time.
            t_w: Waiting time (age of the system).
            t: Observation time (delay after t_w).

        Returns:
            Two-time correlation value.
        """
        T_total, N = trajectory.shape
        if t_w >= T_total or t_w + t >= T_total:
            return np.nan

        phi_tw = trajectory[t_w]
        phi_tw_t = trajectory[t_w + t]
        return float(np.dot(phi_tw_t, phi_tw) / N)

    def compute_response(
        self,
        trajectory: np.ndarray,
        perturbation_trajectory: np.ndarray,
        t_w: int,
        t: int,
    ) -> float:
        """Compute the two-time response function R(t_w + t, t_w).

        R(t_w + t, t_w) = δ<φ(t_w + t)> / δh(t_w)

        Estimated by comparing perturbed and unperturbed trajectories:
        R ≈ [φ_perturbed(t_w + t) - φ_unperturbed(t_w + t)] / h.

        Args:
            trajectory: Unperturbed trajectory, shape (T, N).
            perturbation_trajectory: Perturbed trajectory, shape (T, N).
            t_w: Time at which perturbation was applied.
            t: Observation delay after perturbation.

        Returns:
            Two-time response value.
        """
        T_total, N = trajectory.shape
        if t_w + t >= T_total:
            return np.nan

        delta_phi = perturbation_trajectory[t_w + t] - trajectory[t_w + t]

        # Estimate perturbation strength h from difference at t_w
        h = np.linalg.norm(perturbation_trajectory[t_w] - trajectory[t_w])
        if h < 1e-30:
            return 0.0

        return float(np.mean(delta_phi) / h)

    def aging_scaling(
        self,
        correlations: np.ndarray,
        t_w_values: np.ndarray,
        t_values: np.ndarray,
    ) -> Dict[str, Any]:
        """Test aging scaling C(t_w + t, t_w) = f(t / t_w^μ).

        In simple aging, μ = 1 and correlations depend on t/t_w.
        Sub-aging has μ < 1 and super-aging has μ > 1.

        Args:
            correlations: Array of shape (n_tw, n_t) with C(t_w + t, t_w).
            t_w_values: Waiting times used.
            t_values: Observation times used.

        Returns:
            Dictionary with 'mu' (aging exponent), 'scaling_function'
            (collapsed data), and 'collapse_quality'.
        """
        n_tw, n_t = correlations.shape

        def collapse_error(mu):
            """Measure quality of scaling collapse for a given μ."""
            all_x = []
            all_y = []
            for i, t_w in enumerate(t_w_values):
                if t_w <= 0:
                    continue
                x_scaled = t_values / (t_w ** mu)
                for j in range(n_t):
                    if np.isfinite(correlations[i, j]):
                        all_x.append(x_scaled[j])
                        all_y.append(correlations[i, j])

            if len(all_x) < 5:
                return 1e10

            all_x = np.array(all_x)
            all_y = np.array(all_y)

            # Sort by x and compute variance in sliding windows
            sort_idx = np.argsort(all_x)
            all_x = all_x[sort_idx]
            all_y = all_y[sort_idx]

            window = max(len(all_x) // 20, 3)
            total_var = 0.0
            n_windows = 0
            for k in range(0, len(all_x) - window, window):
                y_window = all_y[k : k + window]
                total_var += np.var(y_window)
                n_windows += 1

            return total_var / max(n_windows, 1)

        # Optimize μ
        result = minimize(collapse_error, x0=1.0, method="Nelder-Mead",
                         options={"xatol": 0.01, "maxiter": 200})
        mu_opt = float(result.x[0])

        # Build collapsed scaling function
        all_x, all_y = [], []
        for i, t_w in enumerate(t_w_values):
            if t_w <= 0:
                continue
            x_scaled = t_values / (t_w ** mu_opt)
            for j in range(n_t):
                if np.isfinite(correlations[i, j]):
                    all_x.append(x_scaled[j])
                    all_y.append(correlations[i, j])

        collapse_quality = 1.0 / (1.0 + result.fun)

        return {
            "mu": mu_opt,
            "scaling_function": (np.array(all_x), np.array(all_y)),
            "collapse_quality": float(collapse_quality),
        }

    def aging_exponent(
        self, correlations: np.ndarray, t_w_values: np.ndarray
    ) -> float:
        """Extract the aging exponent μ from correlation data.

        Uses the decay rate of C at fixed t/t_w ratios across different t_w.

        Args:
            correlations: Array of shape (n_tw, n_t) with C(t_w + t, t_w).
            t_w_values: Waiting times used.

        Returns:
            Aging exponent μ. μ = 1 for full aging, μ < 1 for sub-aging.
        """
        n_tw, n_t = correlations.shape
        if n_tw < 3 or n_t < 2:
            return np.nan

        # Fix t/t_w = 1 and measure how C(2t_w, t_w) varies with t_w
        mid_t_idx = n_t // 2
        c_values = correlations[:, mid_t_idx]
        valid = np.isfinite(c_values) & (t_w_values > 0)

        if valid.sum() < 3:
            return np.nan

        log_tw = np.log(t_w_values[valid])
        log_c = np.log(np.clip(np.abs(c_values[valid]), 1e-30, None))

        coeffs = np.polyfit(log_tw, log_c, 1)
        # C ~ t_w^{-λ/z}, aging exponent μ relates to the h/t_w scaling
        return float(-coeffs[0])

    def time_translation_invariance_test(
        self, correlations: np.ndarray, t_w_values: np.ndarray
    ) -> Dict[str, Any]:
        """Test whether the system is in equilibrium (TTI) or aging.

        Time-translation invariance (TTI): C(t_w + t, t_w) = C(t) depends
        only on the time difference t. Violation of TTI indicates aging.

        Args:
            correlations: Array of shape (n_tw, n_t).
            t_w_values: Waiting times.

        Returns:
            Dictionary with 'is_aging' (bool), 'tti_violation' (scalar
            measuring departure from TTI), and 'p_value'.
        """
        n_tw, n_t = correlations.shape
        if n_tw < 2:
            return {"is_aging": False, "tti_violation": 0.0, "p_value": 1.0}

        # Compare correlations at the same t but different t_w
        variances = np.zeros(n_t)
        for j in range(n_t):
            col = correlations[:, j]
            valid = np.isfinite(col)
            if valid.sum() >= 2:
                variances[j] = np.var(col[valid])

        tti_violation = float(np.mean(variances))
        # Crude significance: compare to expected noise
        overall_var = np.nanvar(correlations)
        threshold = overall_var * 0.1

        is_aging = tti_violation > threshold

        return {
            "is_aging": bool(is_aging),
            "tti_violation": tti_violation,
            "p_value": float(np.exp(-tti_violation / max(threshold, 1e-30))),
        }

    def full_aging_function(
        self, t: float, t_w: float, q_EA: float, tau_0: float
    ) -> float:
        """Evaluate the full aging correlation function.

        C(t, t_w) = q_EA + (1 - q_EA) * h(t / t_w)

        where h(x) = (1 + x)^{-λ/z} is the aging scaling function and
        q_EA is the Edwards-Anderson order parameter (plateau value).

        Args:
            t: Observation time (delay).
            t_w: Waiting time (age).
            q_EA: Edwards-Anderson parameter (long-time plateau).
            tau_0: Microscopic timescale.

        Returns:
            Correlation value C(t_w + t, t_w).
        """
        if t_w < tau_0:
            t_w = tau_0

        x = t / t_w
        # Aging function: power-law decay toward q_EA
        lambda_over_z = 0.5  # typical value
        h_x = (1.0 + x) ** (-lambda_over_z)
        return q_EA + (1.0 - q_EA) * h_x

    def plateau_value(self, correlations: np.ndarray) -> float:
        """Extract the Edwards-Anderson parameter q_EA from correlations.

        q_EA is the plateau value of C(t_w + t, t_w) for intermediate
        times t_0 << t << t_w. It separates the fast (quasi-equilibrium)
        relaxation from the slow (aging) relaxation.

        Args:
            correlations: Array of shape (n_tw, n_t) with two-time correlations.

        Returns:
            Edwards-Anderson order parameter q_EA.
        """
        n_tw, n_t = correlations.shape
        if n_tw == 0 or n_t < 3:
            return 0.0

        # Use the largest t_w, look for plateau in intermediate t
        last_row = correlations[-1]
        valid = np.isfinite(last_row)
        if valid.sum() < 3:
            return 0.0

        c_valid = last_row[valid]
        # Find plateau: region of minimum slope
        grad = np.abs(np.gradient(c_valid))
        if len(grad) < 3:
            return float(np.mean(c_valid))

        # Sliding window to find flattest region
        window = max(len(grad) // 5, 3)
        min_grad = np.inf
        plateau_val = np.mean(c_valid)
        for k in range(len(grad) - window):
            avg_grad = np.mean(grad[k : k + window])
            if avg_grad < min_grad:
                min_grad = avg_grad
                plateau_val = np.mean(c_valid[k : k + window])

        return float(plateau_val)


class FluctuationDissipationViolation:
    """Analysis of fluctuation-dissipation theorem violations.

    In equilibrium, the fluctuation-dissipation theorem (FDT) relates
    the response R and correlation C via T*R = -∂C/∂t_w. Out of
    equilibrium (aging), this is violated: T*R = X * (-∂C/∂t_w)
    where X ≤ 1 is the FDT violation ratio.

    The parametric plot of integrated response χ vs correlation C
    reveals the nature of the aging: a single straight line for
    equilibrium, a broken line for one-step RSB, a continuous curve
    for full RSB.

    Attributes:
        config: AgingConfig with temperature and time parameters.
    """

    def __init__(self, config: AgingConfig):
        """Initialize FDT violation analyzer.

        Args:
            config: Configuration for aging analysis.
        """
        self.config = config

    def fdt_ratio(
        self,
        correlation: np.ndarray,
        response: np.ndarray,
        temperature: float,
    ) -> np.ndarray:
        """Compute the FDT violation ratio X(t, t_w).

        X(t, t_w) = T * R(t, t_w) / (-∂C(t, t_w)/∂t_w)

        In equilibrium X = 1; out of equilibrium X < 1 in general.

        Args:
            correlation: Array of correlation values C(t_w + t, t_w).
            response: Array of response values R(t_w + t, t_w).
            temperature: Effective temperature T.

        Returns:
            Array of FDT violation ratios X.
        """
        dC = np.gradient(correlation)
        # Avoid division by zero
        dC_safe = np.where(np.abs(dC) > 1e-30, dC, np.nan)
        X = temperature * response / (-dC_safe)
        return X

    def fdt_plot(
        self,
        correlations: np.ndarray,
        responses: np.ndarray,
        temperature: float,
    ) -> Dict[str, np.ndarray]:
        """Construct the FDT parametric plot: integrated response χ vs C.

        The integrated response (susceptibility) is
        χ(t, t_w) = ∫_{t_w}^{t_w+t} R(t_w+t, t') dt'.

        In the parametric plot χ(C), equilibrium gives slope -1/T.
        Aging gives a broken or curved line.

        Args:
            correlations: Array of shape (n_tw, n_t).
            responses: Array of shape (n_tw, n_t).
            temperature: Effective temperature T.

        Returns:
            Dictionary with 'C_values' and 'chi_values' for the parametric plot.
        """
        n_tw, n_t = correlations.shape
        C_all = []
        chi_all = []

        for i in range(n_tw):
            c_row = correlations[i]
            r_row = responses[i]
            valid = np.isfinite(c_row) & np.isfinite(r_row)
            if valid.sum() < 3:
                continue

            # Integrated response: cumulative integral of R
            chi = np.zeros(n_t)
            dt = 1.0
            for j in range(1, n_t):
                chi[j] = chi[j - 1] + r_row[j] * dt

            C_all.extend(c_row[valid].tolist())
            chi_all.extend(chi[valid].tolist())

        return {
            "C_values": np.array(C_all),
            "chi_values": np.array(chi_all),
        }

    def effective_temperature(
        self, correlation: np.ndarray, response: np.ndarray
    ) -> np.ndarray:
        """Compute effective temperature T_eff = T / X from FDT violation.

        In the aging regime, different modes thermalize at different
        effective temperatures. Modes with C > q_EA have T_eff = T
        (equilibrated). Modes with C < q_EA have T_eff > T (not equilibrated).

        Args:
            correlation: Correlation values.
            response: Response values.

        Returns:
            Array of effective temperatures.
        """
        dC = np.gradient(correlation)
        # T_eff = -dC / R where both are finite
        mask = np.abs(response) > 1e-30
        T_eff = np.full_like(correlation, np.nan)
        T_eff[mask] = -dC[mask] / response[mask]
        return T_eff

    def x_function(
        self,
        correlations: np.ndarray,
        responses: np.ndarray,
        temperature: float,
    ) -> Callable[[float], float]:
        """Extract the limit function X(C) from the FDT plot.

        X(C) encodes the full replica symmetry breaking structure:
        - X(C) = 1 for all C: equilibrium
        - X(C) = step function: one-step RSB
        - X(C) = continuous: full RSB

        Args:
            correlations: Array of shape (n_tw, n_t).
            responses: Array of shape (n_tw, n_t).
            temperature: Effective temperature.

        Returns:
            Callable X(C) mapping correlation to FDT ratio.
        """
        fdt_data = self.fdt_plot(correlations, responses, temperature)
        C_vals = fdt_data["C_values"]
        chi_vals = fdt_data["chi_values"]

        if len(C_vals) < 3:
            return lambda c: 1.0

        # Sort by C
        sort_idx = np.argsort(C_vals)
        C_sorted = C_vals[sort_idx]
        chi_sorted = chi_vals[sort_idx]

        # X(C) = -T * dχ/dC
        dchi_dC = np.gradient(chi_sorted, C_sorted)
        X_vals = -temperature * dchi_dC

        try:
            X_interp = interp1d(
                C_sorted, X_vals, kind="linear",
                bounds_error=False, fill_value=(X_vals[0], X_vals[-1])
            )
        except ValueError:
            return lambda c: 1.0

        return X_interp

    def one_step_rsb_prediction(
        self, q_EA: float, temperature: float
    ) -> Callable[[float], float]:
        """Predicted X(C) for one-step replica symmetry breaking.

        X(C) = 1 for C > q_EA (quasi-equilibrium regime)
        X(C) = x₁ < 1 for C < q_EA (aging regime)

        This is the simplest non-trivial aging scenario, corresponding
        to a single pair of pure states.

        Args:
            q_EA: Edwards-Anderson parameter.
            temperature: Temperature T.

        Returns:
            Callable X(C) implementing one-step RSB prediction.
        """
        # For one-step RSB, x₁ = T / T_eff where T_eff > T
        x1 = temperature / (temperature + 1.0)  # simplified model

        def X_1rsb(C: float) -> float:
            if C > q_EA:
                return 1.0
            else:
                return x1

        return X_1rsb

    def classify_dynamics(
        self, x_function_values: np.ndarray
    ) -> str:
        """Classify the dynamics from the FDT violation function X(C).

        Categories:
        - 'equilibrium': X ≈ 1 everywhere
        - 'one_step_rsb': X is a step function
        - 'full_rsb': X is a continuous non-trivial function
        - 'non_glassy': X shows no clear pattern

        Args:
            x_function_values: Array of X(C) values evaluated on a grid.

        Returns:
            String classification of the aging dynamics.
        """
        if len(x_function_values) == 0:
            return "non_glassy"

        mean_X = np.nanmean(x_function_values)
        std_X = np.nanstd(x_function_values)

        if mean_X > 0.95 and std_X < 0.05:
            return "equilibrium"

        # Check for step-function behavior
        sorted_X = np.sort(x_function_values[np.isfinite(x_function_values)])
        if len(sorted_X) < 5:
            return "non_glassy"

        # Compute gaps in sorted X values
        diffs = np.diff(sorted_X)
        max_gap = np.max(diffs)
        median_gap = np.median(diffs)

        if max_gap > 10 * median_gap and median_gap > 0:
            return "one_step_rsb"

        if std_X > 0.1:
            return "full_rsb"

        return "non_glassy"

    def training_fdt_analysis(
        self,
        loss_trajectories: List[np.ndarray],
        perturbed_trajectories: List[np.ndarray],
        lr: float,
    ) -> Dict[str, Any]:
        """Perform full FDT analysis on training dynamics.

        Uses multiple independent training runs (unperturbed and perturbed)
        to estimate correlations and responses, then constructs the FDT
        parametric plot.

        Args:
            loss_trajectories: List of 1-D loss arrays from unperturbed runs.
            perturbed_trajectories: List of 1-D loss arrays from perturbed runs.
            lr: Learning rate (used as effective temperature).

        Returns:
            Dictionary with 'fdt_ratio_mean', 'effective_temperature',
            'dynamics_class', and 'fdt_plot_data'.
        """
        n_runs = min(len(loss_trajectories), len(perturbed_trajectories))
        if n_runs == 0:
            return {
                "fdt_ratio_mean": np.nan,
                "effective_temperature": np.nan,
                "dynamics_class": "unknown",
                "fdt_plot_data": {},
            }

        T = len(loss_trajectories[0])
        temperature = lr

        # Average correlation from unperturbed runs
        mean_loss = np.mean([traj for traj in loss_trajectories[:n_runs]], axis=0)
        fluct = np.array([traj - mean_loss for traj in loss_trajectories[:n_runs]])
        auto_corr = np.mean([np.correlate(f, f, mode="full")[T - 1:T + T // 4]
                            for f in fluct], axis=0)
        auto_corr /= (auto_corr[0] + 1e-30)

        # Average response from perturbed runs
        mean_perturbed = np.mean(perturbed_trajectories[:n_runs], axis=0)
        response = (mean_perturbed - mean_loss)[:T // 4 + 1]
        response /= (np.max(np.abs(response)) + 1e-30)

        # FDT ratio
        X = self.fdt_ratio(auto_corr[:len(response)], response, temperature)
        fdt_mean = float(np.nanmean(X))

        T_eff = temperature / max(fdt_mean, 1e-10)

        dynamics_class = self.classify_dynamics(X)

        return {
            "fdt_ratio_mean": fdt_mean,
            "effective_temperature": float(T_eff),
            "dynamics_class": dynamics_class,
            "fdt_plot_data": {
                "correlation": auto_corr[:len(response)],
                "response": response,
            },
        }


class GrokingConnection:
    """Connection between grokking (delayed generalization) and aging phenomena.

    Grokking — where a neural network memorizes training data quickly but
    generalizes much later — can be interpreted through the lens of aging:

    1. The waiting time t_w corresponds to the memorization phase.
    2. The observation time t corresponds to generalization onset.
    3. The two-timescale structure mirrors the fast/slow relaxation in glasses.
    4. The effective temperature may differ in memorization vs generalization.

    This class provides tools to detect grokking, measure aging during
    training, and connect the two phenomena.
    """

    def __init__(self):
        """Initialize grokking-aging connection analyzer."""
        pass

    def detect_grokking(
        self,
        train_losses: np.ndarray,
        test_losses: np.ndarray,
        threshold: float = 0.1,
    ) -> Dict[str, Any]:
        """Detect delayed generalization (grokking).

        Grokking is detected when the training loss reaches a low value
        (memorization) long before the test loss decreases (generalization).

        Args:
            train_losses: Training loss trajectory.
            test_losses: Test loss trajectory.
            threshold: Loss threshold for considering convergence.

        Returns:
            Dictionary with 'is_grokking' (bool), 'memorization_time',
            'generalization_time', 'delay_ratio'.
        """
        # Find memorization time: when train loss first drops below threshold
        mem_idx = np.where(train_losses < threshold)[0]
        mem_time = int(mem_idx[0]) if len(mem_idx) > 0 else len(train_losses)

        # Find generalization time: when test loss first drops below threshold
        gen_idx = np.where(test_losses < threshold)[0]
        gen_time = int(gen_idx[0]) if len(gen_idx) > 0 else len(test_losses)

        delay_ratio = gen_time / max(mem_time, 1)
        is_grokking = delay_ratio > 5.0 and mem_time < len(train_losses)

        return {
            "is_grokking": bool(is_grokking),
            "memorization_time": mem_time,
            "generalization_time": gen_time,
            "delay_ratio": float(delay_ratio),
        }

    def grokking_as_aging(
        self,
        train_trajectory: np.ndarray,
        test_trajectory: np.ndarray,
    ) -> Dict[str, Any]:
        """Interpret grokking through the aging framework.

        Maps grokking phases to aging concepts:
        - Memorization phase → fast β-relaxation (quasi-equilibrium)
        - Plateau phase → aging regime (slow α-relaxation)
        - Generalization onset → escape from metastable state

        Args:
            train_trajectory: Training loss over time.
            test_trajectory: Test loss over time.

        Returns:
            Dictionary with 'phase_boundaries', 'aging_exponent_estimate',
            'plateau_duration', and 'interpretation'.
        """
        grok = self.detect_grokking(train_trajectory, test_trajectory)
        mem_time = grok["memorization_time"]
        gen_time = grok["generalization_time"]

        # Phase boundaries
        phases = {
            "fast_relaxation": (0, mem_time),
            "plateau_aging": (mem_time, gen_time),
            "generalization": (gen_time, len(train_trajectory)),
        }

        # Estimate aging exponent from plateau phase
        if gen_time > mem_time + 10:
            plateau_losses = test_trajectory[mem_time:gen_time]
            if len(plateau_losses) > 5:
                t_arr = np.arange(1, len(plateau_losses) + 1, dtype=float)
                log_t = np.log(t_arr)
                log_l = np.log(np.clip(plateau_losses, 1e-30, None))
                valid = np.isfinite(log_l)
                if valid.sum() >= 3:
                    coeffs = np.polyfit(log_t[valid], log_l[valid], 1)
                    aging_exp = float(-coeffs[0])
                else:
                    aging_exp = np.nan
            else:
                aging_exp = np.nan
        else:
            aging_exp = np.nan

        plateau_duration = gen_time - mem_time

        interpretation = (
            "The grokking delay corresponds to an aging plateau where the "
            "network is trapped in a memorizing metastable state. "
            "Generalization emerges when the system escapes via rare "
            "fluctuations, analogous to α-relaxation in glasses."
        )

        return {
            "phase_boundaries": phases,
            "aging_exponent_estimate": aging_exp,
            "plateau_duration": int(plateau_duration),
            "interpretation": interpretation,
        }

    def two_timescale_dynamics(
        self,
        trajectory: np.ndarray,
        t_mem: int,
        t_gen: int,
    ) -> Dict[str, Any]:
        """Analyze two-timescale dynamics: fast memorization + slow generalization.

        Decomposes the trajectory into a fast component (decaying on
        timescale t_mem) and a slow component (decaying on timescale t_gen).

        Args:
            trajectory: 1-D array (loss or other observable over time).
            t_mem: Memorization timescale.
            t_gen: Generalization timescale.

        Returns:
            Dictionary with 'fast_component', 'slow_component',
            'separation_ratio', and 'fit_quality'.
        """
        T = len(trajectory)
        t_arr = np.arange(T, dtype=float)

        def two_exp(t, A_fast, A_slow, L_inf):
            return A_fast * np.exp(-t / max(t_mem, 1)) + A_slow * np.exp(-t / max(t_gen, 1)) + L_inf

        try:
            popt, pcov = curve_fit(
                two_exp, t_arr, trajectory,
                p0=[trajectory[0] * 0.5, trajectory[0] * 0.5, trajectory[-1]],
                maxfev=5000,
            )
            A_fast, A_slow, L_inf = popt

            fast_component = A_fast * np.exp(-t_arr / max(t_mem, 1))
            slow_component = A_slow * np.exp(-t_arr / max(t_gen, 1))

            predicted = two_exp(t_arr, *popt)
            ss_res = np.sum((trajectory - predicted) ** 2)
            ss_tot = np.sum((trajectory - np.mean(trajectory)) ** 2)
            r_sq = 1.0 - ss_res / (ss_tot + 1e-30)

        except RuntimeError:
            fast_component = np.zeros(T)
            slow_component = trajectory.copy()
            A_fast, A_slow = 0.0, 1.0
            r_sq = 0.0

        return {
            "fast_component": fast_component,
            "slow_component": slow_component,
            "separation_ratio": float(t_gen / max(t_mem, 1)),
            "fit_quality": float(r_sq),
        }

    def correlation_during_grokking(
        self,
        weight_trajectories: np.ndarray,
        t_w_values: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Measure two-time correlations during different grokking phases.

        Computes C(t_w + t, t_w) at several waiting times t_w spanning
        the memorization, plateau, and generalization phases.

        Args:
            weight_trajectories: Array of shape (T, N) with weight snapshots.
            t_w_values: Array of waiting times to probe.

        Returns:
            Dictionary with 'correlations' of shape (n_tw, n_t) and 't_values'.
        """
        T_total, N = weight_trajectories.shape
        max_t = T_total // 4
        t_values = np.unique(np.logspace(0, np.log10(max(max_t, 2)), 30).astype(int))
        n_tw = len(t_w_values)
        n_t = len(t_values)

        correlations = np.full((n_tw, n_t), np.nan)

        for i, t_w in enumerate(t_w_values):
            t_w = int(t_w)
            if t_w >= T_total:
                continue
            phi_tw = weight_trajectories[t_w]
            norm_tw = np.linalg.norm(phi_tw)
            if norm_tw < 1e-30:
                continue

            for j, t in enumerate(t_values):
                t = int(t)
                if t_w + t >= T_total:
                    break
                phi_t = weight_trajectories[t_w + t]
                correlations[i, j] = np.dot(phi_t, phi_tw) / (N * norm_tw ** 2 / N)

        return {"correlations": correlations, "t_values": t_values}

    def effective_temperature_during_grokking(
        self,
        weight_trajectories: np.ndarray,
        gradient_trajectories: np.ndarray,
        lr: float,
    ) -> Dict[str, Any]:
        """Estimate effective temperature during different grokking phases.

        T_eff = lr * <||∇L||²> / (2 * d) where d is the parameter dimension.
        This should differ between memorization and generalization phases.

        Args:
            weight_trajectories: Shape (T, N) weight snapshots.
            gradient_trajectories: Shape (T, N) gradient snapshots.
            lr: Learning rate.

        Returns:
            Dictionary with 'T_eff' array over time, 'mean_T_eff_memorization',
            'mean_T_eff_generalization'.
        """
        T_total, N = gradient_trajectories.shape

        # T_eff(t) = lr * ||∇L(t)||² / (2N)
        grad_norms_sq = np.sum(gradient_trajectories ** 2, axis=1)
        T_eff = lr * grad_norms_sq / (2 * N)

        # Split into early (memorization) and late (generalization)
        midpoint = T_total // 2
        T_eff_mem = float(np.mean(T_eff[:midpoint]))
        T_eff_gen = float(np.mean(T_eff[midpoint:]))

        return {
            "T_eff": T_eff,
            "mean_T_eff_memorization": T_eff_mem,
            "mean_T_eff_generalization": T_eff_gen,
            "temperature_ratio": T_eff_mem / max(T_eff_gen, 1e-30),
        }

    def memory_formation_timescale(
        self, weight_trajectory: np.ndarray
    ) -> float:
        """Estimate when the network forms a stable memory of the training data.

        Measures the decorrelation time of the weight trajectory. Once
        weights stabilize (low velocity), the memory is formed.

        Args:
            weight_trajectory: Array of shape (T, N).

        Returns:
            Estimated memory formation time (training step).
        """
        T, N = weight_trajectory.shape
        velocities = np.diff(weight_trajectory, axis=0)
        speeds = np.linalg.norm(velocities, axis=1)

        # Memory is formed when speed drops below 10% of initial speed
        initial_speed = np.mean(speeds[:10]) if len(speeds) >= 10 else speeds[0]
        threshold = 0.1 * initial_speed

        below_threshold = np.where(speeds < threshold)[0]
        if len(below_threshold) > 0:
            return float(below_threshold[0])
        return float(T)

    def generalization_onset(
        self, test_losses: np.ndarray, threshold: float = 0.5
    ) -> float:
        """Detect when generalization begins (test loss starts decreasing).

        Uses a smoothed version of the test loss to find the onset of
        the generalization phase.

        Args:
            test_losses: Test loss trajectory.
            threshold: Fraction of max test loss below which generalization
                is considered to have begun.

        Returns:
            Estimated generalization onset time.
        """
        # Smooth the test loss
        window = min(len(test_losses) // 20, 50)
        window = max(window, 3)
        kernel = np.ones(window) / window
        smoothed = np.convolve(test_losses, kernel, mode="valid")

        max_loss = np.max(smoothed)
        target = threshold * max_loss

        below = np.where(smoothed < target)[0]
        if len(below) > 0:
            # Find the first sustained drop
            return float(below[0] + window // 2)
        return float(len(test_losses))
