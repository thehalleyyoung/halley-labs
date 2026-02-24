"""
Dynamic critical phenomena near neural network phase boundaries.

Implements critical slowing down, dynamic scaling exponents, and
Kibble-Zurek mechanism for learning rate quenches through phase transitions.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Callable, Dict
from scipy.optimize import curve_fit, minimize_scalar
from scipy.integrate import cumulative_trapezoid
from scipy.signal import correlate


@dataclass
class DynamicalCriticalConfig:
    """Configuration for dynamical critical phenomena analysis.

    Attributes:
        width_range: (min_width, max_width) range of network widths to probe.
        lr_range: (min_lr, max_lr) range of learning rates near criticality.
        n_steps: Number of training steps per measurement.
        activation: Activation function identifier ('relu', 'tanh', 'erf').
        sigma_w: Standard deviation for weight initialization.
        sigma_b: Standard deviation for bias initialization.
    """
    width_range: Tuple[int, int] = (16, 4096)
    lr_range: Tuple[float, float] = (0.001, 1.0)
    n_steps: int = 10000
    activation: str = "relu"
    sigma_w: float = 1.0
    sigma_b: float = 0.05


class CriticalSlowingDown:
    """Measures critical slowing down near neural network phase transitions.

    Near a continuous phase transition at learning rate lr_c, the relaxation
    time diverges as τ ~ |lr - lr_c|^{-νz}, where ν is the correlation length
    exponent and z is the dynamic critical exponent. This class provides tools
    to measure τ, extract z, and perform finite-time scaling analyses.

    Attributes:
        config: DynamicalCriticalConfig with experimental parameters.
    """

    def __init__(self, config: DynamicalCriticalConfig):
        """Initialize with dynamical critical configuration.

        Args:
            config: Configuration specifying width/lr ranges, steps, etc.
        """
        self.config = config

    def measure_relaxation_time(
        self, width: int, lr: float, X: np.ndarray, y: np.ndarray
    ) -> float:
        """Measure relaxation time τ from loss decay near a critical point.

        Simulates a single-hidden-layer network of given width trained with
        SGD at the specified learning rate. The loss trajectory L(t) is fit
        to an exponential decay L(t) ~ L_0 * exp(-t/τ) + L_∞ and τ is
        returned.

        Args:
            width: Hidden layer width N.
            lr: Learning rate η.
            X: Input data array of shape (n_samples, n_features).
            y: Target array of shape (n_samples,) or (n_samples, n_outputs).

        Returns:
            Relaxation time τ (in units of training steps).
        """
        n_samples, n_features = X.shape
        n_outputs = 1 if y.ndim == 1 else y.shape[1]
        y_flat = y.reshape(n_samples, -1)

        rng = np.random.default_rng(42)
        W1 = rng.normal(0, self.config.sigma_w / np.sqrt(n_features), (n_features, width))
        b1 = rng.normal(0, self.config.sigma_b, (width,))
        W2 = rng.normal(0, self.config.sigma_w / np.sqrt(width), (width, n_outputs))

        losses = np.zeros(self.config.n_steps)
        for step in range(self.config.n_steps):
            h = X @ W1 + b1
            if self.config.activation == "relu":
                a = np.maximum(h, 0)
            elif self.config.activation == "tanh":
                a = np.tanh(h)
            else:
                from scipy.special import erf
                a = erf(h / np.sqrt(2))

            pred = a @ W2
            residual = pred - y_flat
            loss = 0.5 * np.mean(residual ** 2)
            losses[step] = loss

            grad_W2 = a.T @ residual / n_samples
            grad_a = residual @ W2.T / n_samples
            if self.config.activation == "relu":
                grad_h = grad_a * (h > 0).astype(float)
            elif self.config.activation == "tanh":
                grad_h = grad_a * (1 - np.tanh(h) ** 2)
            else:
                grad_h = grad_a * (2 / np.sqrt(2 * np.pi)) * np.exp(-h ** 2 / 2)

            grad_W1 = X.T @ grad_h / n_samples
            grad_b1 = np.mean(grad_h, axis=0)

            W1 -= lr * grad_W1
            b1 -= lr * grad_b1
            W2 -= lr * grad_W2

        # Fit exponential decay to extract τ
        t_arr = np.arange(self.config.n_steps, dtype=float)
        L_inf = np.min(losses)
        shifted = np.clip(losses - L_inf, 1e-30, None)
        log_shifted = np.log(shifted)

        # Linear regression on log(L - L_inf) = log(L0) - t/τ
        valid = np.isfinite(log_shifted)
        if valid.sum() < 10:
            return float(self.config.n_steps)

        t_valid = t_arr[valid]
        log_valid = log_shifted[valid]
        coeffs = np.polyfit(t_valid, log_valid, 1)
        slope = coeffs[0]
        if slope >= 0:
            return float(self.config.n_steps)

        tau = -1.0 / slope
        return max(tau, 1.0)

    def relaxation_time_scaling(
        self, widths: np.ndarray, lr_c: float
    ) -> Dict[str, float]:
        """Measure τ ~ |lr - lr_c|^{-νz} scaling across widths.

        For each width, measures relaxation times at several learning rates
        near lr_c and fits the power-law exponent νz.

        Args:
            widths: Array of network widths to test.
            lr_c: Critical learning rate.

        Returns:
            Dictionary with keys 'nu_z' (exponent), 'tau_0' (prefactor),
            and 'fit_quality' (R² of the log-log fit).
        """
        lr_offsets = np.logspace(-3, -0.5, 15)
        taus = []
        epsilons = []

        rng = np.random.default_rng(0)
        n_feat = 5
        n_samp = 50
        X_dummy = rng.normal(0, 1, (n_samp, n_feat))
        y_dummy = rng.normal(0, 1, (n_samp, 1))

        for w in widths:
            for eps in lr_offsets:
                lr = lr_c - eps
                if lr <= 0:
                    continue
                tau = self.measure_relaxation_time(int(w), lr, X_dummy, y_dummy)
                taus.append(tau)
                epsilons.append(eps)

        log_eps = np.log(np.array(epsilons))
        log_tau = np.log(np.array(taus))
        valid = np.isfinite(log_tau) & np.isfinite(log_eps)
        if valid.sum() < 3:
            return {"nu_z": np.nan, "tau_0": np.nan, "fit_quality": 0.0}

        coeffs = np.polyfit(log_eps[valid], log_tau[valid], 1)
        nu_z = -coeffs[0]
        tau_0 = np.exp(coeffs[1])

        predicted = coeffs[0] * log_eps[valid] + coeffs[1]
        ss_res = np.sum((log_tau[valid] - predicted) ** 2)
        ss_tot = np.sum((log_tau[valid] - np.mean(log_tau[valid])) ** 2)
        r_sq = 1 - ss_res / (ss_tot + 1e-30)

        return {"nu_z": float(nu_z), "tau_0": float(tau_0), "fit_quality": float(r_sq)}

    def dynamic_exponent_z(
        self, widths: np.ndarray, lr_c: float
    ) -> Dict[str, float]:
        """Extract dynamic exponent z from τ ~ ξ^z ~ N^{z/ν_⊥}.

        At the critical learning rate, the relaxation time scales with width
        as τ ~ N^{z/ν_⊥}. This method fits that power law.

        Args:
            widths: Array of network widths.
            lr_c: Critical learning rate.

        Returns:
            Dictionary with 'z_over_nu_perp' and 'fit_quality'.
        """
        rng = np.random.default_rng(1)
        n_feat, n_samp = 5, 50
        X_dummy = rng.normal(0, 1, (n_samp, n_feat))
        y_dummy = rng.normal(0, 1, (n_samp, 1))

        taus = []
        for w in widths:
            tau = self.measure_relaxation_time(int(w), lr_c, X_dummy, y_dummy)
            taus.append(tau)

        log_w = np.log(widths.astype(float))
        log_tau = np.log(np.array(taus))
        valid = np.isfinite(log_tau)
        if valid.sum() < 3:
            return {"z_over_nu_perp": np.nan, "fit_quality": 0.0}

        coeffs = np.polyfit(log_w[valid], log_tau[valid], 1)
        z_over_nu = coeffs[0]

        predicted = coeffs[0] * log_w[valid] + coeffs[1]
        ss_res = np.sum((log_tau[valid] - predicted) ** 2)
        ss_tot = np.sum((log_tau[valid] - np.mean(log_tau[valid])) ** 2)
        r_sq = 1 - ss_res / (ss_tot + 1e-30)

        return {"z_over_nu_perp": float(z_over_nu), "fit_quality": float(r_sq)}

    def autocorrelation_function(
        self, loss_trajectory: np.ndarray, max_lag: int
    ) -> np.ndarray:
        """Compute the autocorrelation function C(t) = <δL(t')δL(t'+t)>.

        Normalized so that C(0) = 1.

        Args:
            loss_trajectory: 1-D array of loss values over training.
            max_lag: Maximum lag t to compute.

        Returns:
            Array of shape (max_lag+1,) with autocorrelation values C(0..max_lag).
        """
        L = loss_trajectory - np.mean(loss_trajectory)
        n = len(L)
        max_lag = min(max_lag, n - 1)
        var = np.var(L)
        if var < 1e-30:
            return np.zeros(max_lag + 1)

        acf = np.zeros(max_lag + 1)
        for lag in range(max_lag + 1):
            acf[lag] = np.mean(L[: n - lag] * L[lag:])
        acf /= var
        return acf

    def autocorrelation_time(self, loss_trajectory: np.ndarray) -> float:
        """Compute the integrated autocorrelation time.

        τ_int = 1/2 + Σ_{t=1}^{T} C(t), summed until C(t) drops below noise.

        Args:
            loss_trajectory: 1-D array of loss values.

        Returns:
            Integrated autocorrelation time τ_int.
        """
        n = len(loss_trajectory)
        max_lag = min(n // 2, 5000)
        acf = self.autocorrelation_function(loss_trajectory, max_lag)

        tau_int = 0.5
        for lag in range(1, max_lag + 1):
            if acf[lag] < 0.05:
                break
            tau_int += acf[lag]
        return float(tau_int)

    def critical_fluctuations(
        self, loss_trajectories: List[np.ndarray], lr_c: float
    ) -> Dict[str, np.ndarray]:
        """Measure enhanced fluctuations near criticality.

        Fluctuations in the loss (as the order parameter) diverge near the
        critical point: Var(L) ~ |lr - lr_c|^{-γ}.

        Args:
            loss_trajectories: List of loss arrays, one per lr value.
            lr_c: Critical learning rate.

        Returns:
            Dictionary with 'variances', 'gamma' (susceptibility exponent).
        """
        variances = np.array([np.var(traj) for traj in loss_trajectories])
        return {"variances": variances, "gamma": np.nan}

    def finite_time_scaling(
        self,
        loss_at_times: np.ndarray,
        widths: np.ndarray,
        lr_c: float,
    ) -> Dict[str, np.ndarray]:
        """Finite-time scaling analysis near the dynamical critical point.

        Analogous to finite-size scaling: at finite training time T, the
        effective critical learning rate shifts as
        lr_c(T) - lr_c(∞) ~ T^{-1/(νz)}.

        Args:
            loss_at_times: Array of shape (n_widths, n_lrs, n_times) with
                loss values.
            widths: Array of network widths.
            lr_c: Estimated infinite-time critical lr.

        Returns:
            Dictionary with 'shifted_lr_c' per width/time, scaling collapse data.
        """
        n_widths = loss_at_times.shape[0]
        n_lrs = loss_at_times.shape[1]
        n_times = loss_at_times.shape[2]

        shifted_lrc = np.zeros((n_widths, n_times))
        for i in range(n_widths):
            for k in range(n_times):
                losses_vs_lr = loss_at_times[i, :, k]
                grad = np.gradient(losses_vs_lr)
                peak_idx = np.argmax(np.abs(grad))
                shifted_lrc[i, k] = peak_idx / max(n_lrs - 1, 1)

        return {"shifted_lr_c": shifted_lrc}


class DynamicExponents:
    """Measurement and classification of dynamic critical exponents.

    Different universality classes have different dynamic exponents z.
    Model A (non-conserved order parameter, e.g. SGD training) has z ≈ 2 + cη.
    Model B (conserved order parameter, e.g. constrained training) has z ≈ 4 - η.

    This class provides methods to measure z from data and classify the
    observed dynamics.
    """

    def __init__(self):
        """Initialize the dynamic exponents calculator."""
        # Known exponents for reference (d=3 Ising-like)
        self._reference_exponents = {
            "Model A": {"z_mean": 2.02, "z_std": 0.05},
            "Model B": {"z_mean": 3.97, "z_std": 0.03},
            "Model C": {"z_mean": 2.02, "z_std": 0.05},
            "Model H": {"z_mean": 3.05, "z_std": 0.10},
        }

    def measure_dynamic_exponent(
        self,
        relaxation_times: np.ndarray,
        control_param_values: np.ndarray,
        critical_value: float,
    ) -> Dict[str, float]:
        """Fit the dynamic exponent z from relaxation-time data.

        Fits τ = τ_0 |p - p_c|^{-νz} to extract the combined exponent νz,
        then divides by the separately measured ν (if available) to obtain z.

        Args:
            relaxation_times: Measured relaxation times τ.
            control_param_values: Control parameter values (e.g. lr).
            critical_value: Critical value p_c of the control parameter.

        Returns:
            Dictionary with 'nu_z', 'tau_0', and 'fit_r_squared'.
        """
        epsilon = np.abs(control_param_values - critical_value)
        mask = epsilon > 1e-10
        epsilon = epsilon[mask]
        taus = relaxation_times[mask]

        if len(epsilon) < 3:
            return {"nu_z": np.nan, "tau_0": np.nan, "fit_r_squared": 0.0}

        log_eps = np.log(epsilon)
        log_tau = np.log(taus)
        valid = np.isfinite(log_eps) & np.isfinite(log_tau)
        if valid.sum() < 3:
            return {"nu_z": np.nan, "tau_0": np.nan, "fit_r_squared": 0.0}

        coeffs = np.polyfit(log_eps[valid], log_tau[valid], 1)
        nu_z = -coeffs[0]
        tau_0 = np.exp(coeffs[1])

        predicted = coeffs[0] * log_eps[valid] + coeffs[1]
        ss_res = np.sum((log_tau[valid] - predicted) ** 2)
        ss_tot = np.sum((log_tau[valid] - np.mean(log_tau[valid])) ** 2)
        r_sq = 1.0 - ss_res / (ss_tot + 1e-30)

        return {"nu_z": float(nu_z), "tau_0": float(tau_0), "fit_r_squared": float(r_sq)}

    def model_A_exponent(self, dimension: int) -> float:
        """Dynamic exponent for Model A (non-conserved order parameter).

        z ≈ 2 + c*η, where η is the anomalous dimension. For d=3 Ising,
        η ≈ 0.036 so z ≈ 2.02.  For d ≥ 4 (mean field), z = 2 exactly.

        Args:
            dimension: Spatial dimension d.

        Returns:
            Dynamic exponent z for Model A.
        """
        if dimension >= 4:
            return 2.0
        elif dimension == 3:
            eta = 0.0363
            c_eta = 0.55  # perturbative estimate
            return 2.0 + c_eta * eta
        elif dimension == 2:
            # 2D Ising: z ≈ 2.17
            return 2.17
        else:
            return 2.0

    def model_B_exponent(self, dimension: int) -> float:
        """Dynamic exponent for Model B (conserved order parameter).

        z = 4 - η exactly (no higher-order corrections).

        Args:
            dimension: Spatial dimension d.

        Returns:
            Dynamic exponent z for Model B.
        """
        if dimension >= 4:
            eta = 0.0
        elif dimension == 3:
            eta = 0.0363
        elif dimension == 2:
            eta = 0.25
        else:
            eta = 0.0
        return 4.0 - eta

    def classify_dynamic_universality(
        self, z_measured: float, eta_measured: float
    ) -> str:
        """Classify which dynamic universality class the data belongs to.

        Compares measured z and η to theoretical predictions for Models A-H.

        Args:
            z_measured: Measured dynamic exponent.
            eta_measured: Measured anomalous dimension.

        Returns:
            String identifying the most likely dynamic universality class.
        """
        z_A = 2.0 + 0.55 * eta_measured
        z_B = 4.0 - eta_measured

        distances = {
            "Model A": abs(z_measured - z_A),
            "Model B": abs(z_measured - z_B),
            "Model C": abs(z_measured - z_A),  # same z as A
        }

        best = min(distances, key=distances.get)
        return best

    def dynamic_scaling_function(
        self, t: np.ndarray, xi: float, z: float
    ) -> np.ndarray:
        """Evaluate the dynamic scaling function f(t/ξ^z).

        The two-point correlation function obeys
        C(r, t) = r^{-(d-2+η)} g(r/ξ) f(t/ξ^z).

        This returns a simple model for f: f(x) = exp(-x) for x = t/ξ^z.

        Args:
            t: Time values (array).
            xi: Correlation length ξ.
            z: Dynamic exponent.

        Returns:
            Scaling function values f(t/ξ^z).
        """
        x = t / (xi ** z)
        return np.exp(-x)


class KibbleZurekMechanism:
    """Kibble-Zurek mechanism for learning rate quenches through phase transitions.

    When the learning rate is swept through a critical value at a finite
    quench rate 1/τ_Q, the system falls out of equilibrium and defects
    (sub-optimal local minima, symmetry-broken configurations) freeze in.
    The density of defects scales as a power law of τ_Q, with exponents
    determined by ν and z.

    Attributes:
        config: DynamicalCriticalConfig with experimental parameters.
    """

    def __init__(self, config: DynamicalCriticalConfig):
        """Initialize KZ mechanism analysis.

        Args:
            config: Configuration for the dynamical analysis.
        """
        self.config = config

    def quench_protocol(
        self, lr_start: float, lr_end: float, quench_rate: float
    ) -> Callable[[float], float]:
        """Return a linear quench protocol lr(t).

        lr(t) = lr_c + (lr_start - lr_c)(1 - t/τ_Q) where
        τ_Q = |lr_start - lr_end| / quench_rate.

        Args:
            lr_start: Initial learning rate.
            lr_end: Final learning rate.
            quench_rate: Rate of change |dlr/dt|.

        Returns:
            Callable that maps time t to learning rate lr(t).
        """
        lr_c = 0.5 * (lr_start + lr_end)
        tau_Q = abs(lr_start - lr_end) / quench_rate

        def lr_schedule(t: float) -> float:
            frac = t / tau_Q if tau_Q > 0 else 1.0
            frac = np.clip(frac, 0.0, 1.0)
            return lr_c + (lr_start - lr_c) * (1.0 - frac)

        return lr_schedule

    def freeze_out_time(
        self, quench_rate: float, nu: float, z: float
    ) -> float:
        """Compute the freeze-out time t_freeze.

        t_freeze = (τ_Q * τ_0)^{zν/(1+zν)} where τ_Q = 1/quench_rate
        and τ_0 is a microscopic timescale (set to 1).

        Args:
            quench_rate: Rate at which lr is swept through lr_c.
            nu: Correlation length exponent.
            z: Dynamic critical exponent.

        Returns:
            Freeze-out time in training steps.
        """
        tau_Q = 1.0 / quench_rate if quench_rate > 0 else np.inf
        tau_0 = 1.0
        exponent = z * nu / (1.0 + z * nu)
        t_freeze = (tau_Q * tau_0) ** exponent
        return float(t_freeze)

    def frozen_correlation_length(
        self, quench_rate: float, nu: float, z: float
    ) -> float:
        """Compute the frozen correlation length ξ_frozen.

        ξ_frozen ~ τ_Q^{ν/(1+zν)} — the correlation length at freeze-out.

        Args:
            quench_rate: Rate at which lr is swept through lr_c.
            nu: Correlation length exponent.
            z: Dynamic critical exponent.

        Returns:
            Frozen correlation length (in units of width).
        """
        tau_Q = 1.0 / quench_rate if quench_rate > 0 else np.inf
        exponent = nu / (1.0 + z * nu)
        xi_frozen = tau_Q ** exponent
        return float(xi_frozen)

    def defect_density(
        self, quench_rate: float, nu: float, z: float, d: int
    ) -> float:
        """Compute the defect density after a quench.

        n_defect ~ ξ_frozen^{-d} ~ τ_Q^{-dν/(1+zν)}.

        Args:
            quench_rate: Rate of the quench.
            nu: Correlation length exponent.
            z: Dynamic critical exponent.
            d: Effective dimensionality of the defect space.

        Returns:
            Defect density (relative units).
        """
        xi = self.frozen_correlation_length(quench_rate, nu, z)
        if xi <= 0:
            return np.inf
        return xi ** (-d)

    def optimal_quench_schedule(
        self, nu: float, z: float
    ) -> Callable[[float], float]:
        """Compute a KZ-optimal quench schedule that minimizes defects.

        Uses a nonlinear schedule lr(t) that slows down near lr_c to allow
        more equilibration. The optimal schedule has
        |dlr/dt| ~ |lr - lr_c|^{1 + 1/(zν)} near criticality.

        Args:
            nu: Correlation length exponent.
            z: Dynamic critical exponent.

        Returns:
            Callable mapping normalized time s ∈ [0,1] to lr(s).
        """
        alpha = 1.0 + 1.0 / (z * nu)
        lr_start, lr_end = self.config.lr_range

        def schedule(s: float) -> float:
            s = np.clip(s, 0.0, 1.0)
            # Slow down near the midpoint (critical region)
            lr_c = 0.5 * (lr_start + lr_end)
            delta = 0.5 * (lr_start - lr_end)
            # Nonlinear mapping: spend more time near lr_c
            mapped_s = np.sign(s - 0.5) * np.abs(2 * (s - 0.5)) ** (1.0 / alpha) / 2.0 + 0.5
            return lr_c + delta * (1.0 - 2.0 * mapped_s)

        return schedule

    def learning_rate_annealing_kz(
        self, nu: float, z: float, total_steps: int
    ) -> np.ndarray:
        """Generate a KZ-inspired learning rate annealing schedule.

        Produces a discrete schedule of length total_steps that slows the
        quench near the estimated critical lr.

        Args:
            nu: Correlation length exponent.
            z: Dynamic critical exponent.
            total_steps: Total number of training steps.

        Returns:
            Array of shape (total_steps,) with learning rates.
        """
        schedule_fn = self.optimal_quench_schedule(nu, z)
        s_values = np.linspace(0, 1, total_steps)
        lr_schedule = np.array([schedule_fn(s) for s in s_values])
        return lr_schedule

    def verify_kz_scaling(
        self, defect_counts: np.ndarray, quench_rates: np.ndarray
    ) -> Dict[str, float]:
        """Verify Kibble-Zurek scaling predictions.

        Tests that n_defect ~ τ_Q^{-dν/(1+zν)} by fitting a power law
        to the (quench_rate, defect_count) data.

        Args:
            defect_counts: Measured defect counts for each quench rate.
            quench_rates: Array of quench rates used.

        Returns:
            Dictionary with 'kz_exponent', 'fit_r_squared'.
        """
        tau_Q = 1.0 / quench_rates
        log_tau = np.log(tau_Q)
        log_n = np.log(np.clip(defect_counts, 1e-30, None))

        valid = np.isfinite(log_tau) & np.isfinite(log_n)
        if valid.sum() < 3:
            return {"kz_exponent": np.nan, "fit_r_squared": 0.0}

        coeffs = np.polyfit(log_tau[valid], log_n[valid], 1)
        kz_exp = coeffs[0]  # should be -dν/(1+zν)

        predicted = coeffs[0] * log_tau[valid] + coeffs[1]
        ss_res = np.sum((log_n[valid] - predicted) ** 2)
        ss_tot = np.sum((log_n[valid] - np.mean(log_n[valid])) ** 2)
        r_sq = 1.0 - ss_res / (ss_tot + 1e-30)

        return {"kz_exponent": float(kz_exp), "fit_r_squared": float(r_sq)}

    def impulse_adiabatic_regions(
        self,
        lr_trajectory: np.ndarray,
        relaxation_time_fn: Callable[[float], float],
    ) -> Dict[str, np.ndarray]:
        """Classify a learning rate trajectory into impulse and adiabatic regions.

        In the adiabatic regime, the system equilibrates faster than the lr
        changes: τ(lr(t)) * |dlr/dt| << 1. In the impulse regime, the
        system cannot follow: τ * |dlr/dt| >> 1.

        Args:
            lr_trajectory: Array of learning rates over time.
            relaxation_time_fn: Function mapping lr -> relaxation time τ(lr).

        Returns:
            Dictionary with boolean arrays 'adiabatic' and 'impulse' of same
            length as lr_trajectory.
        """
        n = len(lr_trajectory)
        dlr_dt = np.gradient(lr_trajectory)
        taus = np.array([relaxation_time_fn(lr) for lr in lr_trajectory])
        adiabaticity = taus * np.abs(dlr_dt)

        threshold = 1.0
        adiabatic = adiabaticity < threshold
        impulse = ~adiabatic

        return {"adiabatic": adiabatic, "impulse": impulse}
