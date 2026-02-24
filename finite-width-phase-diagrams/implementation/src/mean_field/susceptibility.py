"""
Susceptibility computation for neural network mean-field theory.

Implements linear response, fluctuation-dissipation relations,
dynamic susceptibility, critical exponent extraction, and finite-size scaling.
"""

import numpy as np
from scipy import integrate, optimize, interpolate, signal
from scipy.fft import fft, fftfreq


class LinearResponseComputer:
    """Linear response theory computations for mean-field systems."""

    def __init__(self, system_size, temperature=1.0):
        self.system_size = system_size
        self.temperature = temperature
        self.beta = 1.0 / temperature

    def compute_static_susceptibility(self, correlation_matrix):
        """
        Compute static susceptibility from correlation matrix.
        χ = β * ⟨δm²⟩ where δm = m - ⟨m⟩ is the fluctuation.

        Parameters
        ----------
        correlation_matrix : ndarray, shape (N, N)
            Connected correlation matrix C_ij = ⟨δs_i δs_j⟩.

        Returns
        -------
        chi : float
            Static susceptibility per site.
        """
        N = correlation_matrix.shape[0]
        total_variance = np.sum(correlation_matrix)
        chi = self.beta * total_variance / N
        return chi

    def connected_susceptibility(self, magnetizations, fields):
        """
        Compute connected susceptibility via numerical derivative.
        χ_conn = ∂⟨m⟩/∂h evaluated by finite differences.

        Parameters
        ----------
        magnetizations : ndarray, shape (n_fields,)
            Average magnetization at each field value.
        fields : ndarray, shape (n_fields,)
            Applied field values (must be sorted).

        Returns
        -------
        chi_conn : ndarray, shape (n_fields,)
            Connected susceptibility at each field value.
        """
        n = len(fields)
        chi_conn = np.zeros(n)
        # Central differences in interior, forward/backward at boundaries
        for i in range(n):
            if i == 0:
                chi_conn[i] = (magnetizations[1] - magnetizations[0]) / (fields[1] - fields[0])
            elif i == n - 1:
                chi_conn[i] = (magnetizations[-1] - magnetizations[-2]) / (fields[-1] - fields[-2])
            else:
                dh_fwd = fields[i + 1] - fields[i]
                dh_bwd = fields[i] - fields[i - 1]
                # Second-order central difference for non-uniform spacing
                chi_conn[i] = (
                    magnetizations[i + 1] * dh_bwd / dh_fwd
                    - magnetizations[i - 1] * dh_fwd / dh_bwd
                    + magnetizations[i] * (dh_fwd - dh_bwd) / (dh_fwd * dh_bwd)
                ) / (dh_fwd + dh_bwd) * 2.0
        return chi_conn

    def susceptibility_tensor(self, order_params, perturbation_directions):
        """
        Compute the full susceptibility tensor χ_ij = ∂⟨O_i⟩/∂h_j.

        Parameters
        ----------
        order_params : ndarray, shape (n_perturbations, n_observables)
            Order parameter values for each perturbation applied.
            Row k corresponds to perturbation along direction k.
        perturbation_directions : ndarray, shape (n_perturbations, n_directions)
            Perturbation vectors applied. Each row is a direction with magnitude = field strength.

        Returns
        -------
        chi_tensor : ndarray, shape (n_observables, n_directions)
            Susceptibility tensor.
        """
        # Solve χ · h = δO via least squares: each perturbation gives one equation
        # order_params[k, i] = sum_j chi[i, j] * perturbation_directions[k, j]
        n_obs = order_params.shape[1]
        n_dir = perturbation_directions.shape[1]
        chi_tensor = np.zeros((n_obs, n_dir))

        for i in range(n_obs):
            # Solve: perturbation_directions @ chi[i, :] = order_params[:, i]
            result, _, _, _ = np.linalg.lstsq(perturbation_directions, order_params[:, i], rcond=None)
            chi_tensor[i, :] = result

        return chi_tensor

    def kramers_kronig_check(self, chi_real, chi_imag, frequencies):
        """
        Verify Kramers-Kronig relations between real and imaginary parts.

        χ'(ω) = (1/π) P∫ χ''(ω') / (ω' - ω) dω'
        χ''(ω) = -(1/π) P∫ χ'(ω') / (ω' - ω) dω'

        Parameters
        ----------
        chi_real : ndarray
            Real part χ'(ω).
        chi_imag : ndarray
            Imaginary part χ''(ω).
        frequencies : ndarray
            Angular frequencies ω.

        Returns
        -------
        result : dict
            'chi_real_from_imag': χ' reconstructed from χ'' via KK.
            'chi_imag_from_real': χ'' reconstructed from χ' via KK.
            'real_error': relative L2 error in χ' reconstruction.
            'imag_error': relative L2 error in χ'' reconstruction.
            'consistent': bool, True if both errors < 0.1.
        """
        n = len(frequencies)
        dw = np.diff(frequencies)
        chi_real_reconstructed = np.zeros(n)
        chi_imag_reconstructed = np.zeros(n)

        for i in range(n):
            omega = frequencies[i]
            # Principal value integral via excluding the pole
            mask = np.abs(frequencies - omega) > 1e-10 * (np.max(frequencies) - np.min(frequencies))
            if np.sum(mask) < 2:
                continue
            denom = frequencies[mask] - omega
            # χ'(ω) from χ''
            integrand_real = chi_imag[mask] / denom
            chi_real_reconstructed[i] = (1.0 / np.pi) * np.trapz(integrand_real, frequencies[mask])
            # χ''(ω) from χ'
            integrand_imag = chi_real[mask] / denom
            chi_imag_reconstructed[i] = -(1.0 / np.pi) * np.trapz(integrand_imag, frequencies[mask])

        norm_real = np.linalg.norm(chi_real)
        norm_imag = np.linalg.norm(chi_imag)
        real_error = np.linalg.norm(chi_real - chi_real_reconstructed) / max(norm_real, 1e-15)
        imag_error = np.linalg.norm(chi_imag - chi_imag_reconstructed) / max(norm_imag, 1e-15)

        return {
            "chi_real_from_imag": chi_real_reconstructed,
            "chi_imag_from_real": chi_imag_reconstructed,
            "real_error": real_error,
            "imag_error": imag_error,
            "consistent": bool(real_error < 0.1 and imag_error < 0.1),
        }

    def sum_rule_check(self, susceptibility_vs_freq):
        """
        Verify frequency sum rules for susceptibility.

        Checks: (1/π) ∫₀^∞ χ''(ω)/ω dω = χ_static  (first moment sum rule)
                (2/π) ∫₀^∞ ω·χ''(ω) dω = -⟨[A,[H,A]]⟩  (f-sum rule proxy)

        Parameters
        ----------
        susceptibility_vs_freq : dict
            'chi_real': ndarray of χ'(ω).
            'chi_imag': ndarray of χ''(ω).
            'frequencies': ndarray of ω values (positive).

        Returns
        -------
        result : dict
            'static_from_sum_rule': χ_static from integral of χ''/ω.
            'static_from_zero_freq': χ'(ω→0).
            'relative_error': relative difference.
            'first_moment': ∫ χ''(ω)/ω dω / π.
            'second_moment': ∫ ω·χ''(ω) dω * 2/π.
        """
        freqs = susceptibility_vs_freq["frequencies"]
        chi_real = susceptibility_vs_freq["chi_real"]
        chi_imag = susceptibility_vs_freq["chi_imag"]

        positive = freqs > 0
        w_pos = freqs[positive]
        chi_imag_pos = chi_imag[positive]
        chi_real_pos = chi_real[positive]

        # First moment sum rule
        integrand_1 = chi_imag_pos / w_pos
        first_moment = np.trapz(integrand_1, w_pos) / np.pi

        # Second moment
        integrand_2 = w_pos * chi_imag_pos
        second_moment = 2.0 * np.trapz(integrand_2, w_pos) / np.pi

        # Static susceptibility from zero-frequency limit
        idx_min = np.argmin(np.abs(freqs))
        chi_static_direct = chi_real[idx_min]

        rel_err = abs(first_moment - chi_static_direct) / max(abs(chi_static_direct), 1e-15)

        return {
            "static_from_sum_rule": first_moment,
            "static_from_zero_freq": chi_static_direct,
            "relative_error": rel_err,
            "first_moment": first_moment,
            "second_moment": second_moment,
        }


class FluctuationDissipation:
    """Fluctuation-dissipation relations and aging analysis."""

    def __init__(self, temperature=1.0):
        self.temperature = temperature

    def verify_fdt(self, correlation, response, times):
        """
        Check fluctuation-dissipation theorem in equilibrium.
        In equilibrium: R(t-t') = -(β/1) * θ(t-t') * dC(t-t')/d(t-t')

        Parameters
        ----------
        correlation : ndarray, shape (n_times,)
            C(τ) = ⟨A(t+τ)A(t)⟩ as a function of time difference τ.
        response : ndarray, shape (n_times,)
            R(τ) = response function for τ ≥ 0.
        times : ndarray, shape (n_times,)
            Time differences τ ≥ 0.

        Returns
        -------
        result : dict
            'holds': bool, whether FDT is satisfied to tolerance.
            'max_violation': maximum absolute violation.
            'violation_profile': pointwise violation R(τ) + β·dC/dτ.
            'relative_violation': relative L2 norm of violation.
        """
        dt = np.diff(times)
        dC_dt = np.zeros_like(correlation)
        # Central differences
        dC_dt[1:-1] = (correlation[2:] - correlation[:-2]) / (times[2:] - times[:-2])
        dC_dt[0] = (correlation[1] - correlation[0]) / dt[0]
        dC_dt[-1] = (correlation[-1] - correlation[-2]) / dt[-1]

        beta = 1.0 / self.temperature
        # FDT: R(τ) = -β * dC/dτ for τ > 0
        fdt_prediction = -beta * dC_dt
        violation = response - fdt_prediction

        norm_response = np.linalg.norm(response)
        rel_violation = np.linalg.norm(violation) / max(norm_response, 1e-15)

        return {
            "holds": bool(rel_violation < 0.05),
            "max_violation": float(np.max(np.abs(violation))),
            "violation_profile": violation,
            "relative_violation": float(rel_violation),
        }

    def effective_temperature(self, correlation, response, times):
        """
        Extract effective temperature from FDT violation.
        T_eff(τ) = -dC/dτ / R(τ) when FDT is violated.

        Parameters
        ----------
        correlation : ndarray, shape (n_times,)
            Two-time correlation function C(τ).
        response : ndarray, shape (n_times,)
            Response function R(τ).
        times : ndarray, shape (n_times,)
            Time differences τ.

        Returns
        -------
        result : dict
            'T_eff': ndarray of effective temperature vs τ.
            'T_eff_mean': mean effective temperature.
            'T_eff_std': standard deviation.
            'times': time values used (excluding problematic points).
        """
        dC_dt = np.gradient(correlation, times)

        # Avoid division by zero in response
        valid = np.abs(response) > 1e-15
        T_eff = np.full_like(correlation, np.nan)
        T_eff[valid] = -dC_dt[valid] / response[valid]

        finite_mask = np.isfinite(T_eff) & valid
        T_eff_finite = T_eff[finite_mask]

        return {
            "T_eff": T_eff,
            "T_eff_mean": float(np.mean(T_eff_finite)) if len(T_eff_finite) > 0 else np.nan,
            "T_eff_std": float(np.std(T_eff_finite)) if len(T_eff_finite) > 0 else np.nan,
            "times": times[finite_mask],
        }

    def fdt_ratio(self, correlation, response):
        """
        Compute the FDT ratio X = T·R/(-dC/dt_w) measuring equilibrium deviation.
        X = 1 in equilibrium, X < 1 indicates aging/out-of-equilibrium.

        Parameters
        ----------
        correlation : ndarray, shape (n_tw, n_t)
            C(t, t_w) two-time correlation for each waiting time.
        response : ndarray, shape (n_tw, n_t)
            R(t, t_w) two-time response for each waiting time.

        Returns
        -------
        X : ndarray, shape (n_tw, n_t)
            FDT ratio at each (t, t_w).
        """
        n_tw, n_t = correlation.shape
        X = np.full((n_tw, n_t), np.nan)

        for i in range(n_tw):
            dC = np.gradient(correlation[i, :])
            valid = np.abs(dC) > 1e-15
            X[i, valid] = -self.temperature * response[i, valid] / dC[valid]

        return X

    def parametric_fdt_plot(self, correlation, response):
        """
        Construct parametric FDT plot: integrated response χ(t,tw) vs C(t,tw).
        In equilibrium, slope = -1/T. Deviations indicate aging.

        Parameters
        ----------
        correlation : ndarray, shape (n_times,)
            C(t, t_w) for fixed t_w as function of t.
        response : ndarray, shape (n_times,)
            R(t, t_w) for fixed t_w as function of t.

        Returns
        -------
        result : dict
            'C_values': correlation values (x-axis of parametric plot).
            'chi_values': integrated response (y-axis).
            'slope_equilibrium': expected slope -1/T.
            'measured_slopes': local slopes dχ/dC.
            'X_fdt': local FDT ratio X = -T * dχ/dC.
        """
        # Integrate response to get susceptibility χ(t) = ∫₀ᵗ R(t') dt'
        n = len(response)
        chi_values = np.zeros(n)
        for i in range(1, n):
            chi_values[i] = chi_values[i - 1] + 0.5 * (response[i] + response[i - 1])

        # Sort by correlation value for parametric plot
        sort_idx = np.argsort(correlation)
        C_sorted = correlation[sort_idx]
        chi_sorted = chi_values[sort_idx]

        # Local slopes
        slopes = np.gradient(chi_sorted, C_sorted)
        X_fdt = -self.temperature * slopes

        return {
            "C_values": C_sorted,
            "chi_values": chi_sorted,
            "slope_equilibrium": -1.0 / self.temperature,
            "measured_slopes": slopes,
            "X_fdt": X_fdt,
        }

    def aging_analysis(self, two_time_data, waiting_times):
        """
        Analyze aging behavior from FDT violations across waiting times.

        Parameters
        ----------
        two_time_data : list of dict
            Each element has keys 'correlation', 'response', 'times'
            for a given waiting time t_w.
        waiting_times : ndarray
            The waiting times t_w.

        Returns
        -------
        result : dict
            'T_eff_vs_tw': effective temperature for each t_w.
            'X_vs_tw': average FDT ratio for each t_w.
            'aging_exponent': μ from T_eff ~ t_w^μ fit.
            'is_aging': whether system shows aging (X < 1 systematically).
        """
        n_tw = len(waiting_times)
        T_eff_arr = np.zeros(n_tw)
        X_arr = np.zeros(n_tw)

        for i, data in enumerate(two_time_data):
            corr = data["correlation"]
            resp = data["response"]
            t = data["times"]

            eff = self.effective_temperature(corr, resp, t)
            T_eff_arr[i] = eff["T_eff_mean"]

            dC = np.gradient(corr, t)
            valid = np.abs(dC) > 1e-15
            if np.sum(valid) > 0:
                X_local = -self.temperature * resp[valid] / dC[valid]
                X_arr[i] = np.mean(X_local[np.isfinite(X_local)])
            else:
                X_arr[i] = np.nan

        # Fit aging exponent: T_eff ~ t_w^μ
        valid_tw = (waiting_times > 0) & np.isfinite(T_eff_arr) & (T_eff_arr > 0)
        aging_exponent = 0.0
        if np.sum(valid_tw) >= 2:
            log_tw = np.log(waiting_times[valid_tw])
            log_T = np.log(T_eff_arr[valid_tw])
            coeffs = np.polyfit(log_tw, log_T, 1)
            aging_exponent = coeffs[0]

        is_aging = bool(np.nanmean(X_arr) < 0.95)

        return {
            "T_eff_vs_tw": T_eff_arr,
            "X_vs_tw": X_arr,
            "aging_exponent": float(aging_exponent),
            "is_aging": is_aging,
        }


class DynamicSusceptibility:
    """Frequency-dependent susceptibility χ(ω) computations."""

    def __init__(self, max_frequency=100.0, n_frequencies=500):
        self.max_frequency = max_frequency
        self.n_frequencies = n_frequencies
        self.frequencies = np.linspace(0, max_frequency, n_frequencies)

    def compute_chi_omega(self, correlation_time_series, dt):
        """
        Compute frequency-dependent susceptibility χ(ω) via Fourier transform.
        χ(ω) = β[C(0) - iω C̃(ω)] where C̃ is the Fourier transform of C(t).

        Parameters
        ----------
        correlation_time_series : ndarray
            Time-domain correlation function C(t) for t ≥ 0.
        dt : float
            Time step between samples.

        Returns
        -------
        result : dict
            'chi_omega': complex susceptibility χ(ω).
            'frequencies': angular frequencies ω.
        """
        n = len(correlation_time_series)
        # Apply window to reduce spectral leakage
        window = signal.windows.hann(2 * n)[:n]
        windowed = correlation_time_series * window

        # One-sided Fourier transform (since C(t) defined for t >= 0)
        freqs = fftfreq(2 * n, d=dt)[:n]
        omega = 2.0 * np.pi * freqs

        # Numerical Fourier transform: C̃(ω) = ∫₀^∞ C(t) e^{-iωt} dt
        C_tilde = np.zeros(len(omega), dtype=complex)
        t = np.arange(n) * dt
        for k in range(len(omega)):
            integrand = windowed * np.exp(-1j * omega[k] * t)
            C_tilde[k] = np.trapz(integrand, t)

        # χ(ω) = 1 - iω·C̃(ω) / C(0) in normalized form
        C0 = correlation_time_series[0]
        if abs(C0) > 1e-15:
            chi_omega = (C0 - 1j * omega * C_tilde) / (self.max_frequency * dt)
        else:
            chi_omega = -1j * omega * C_tilde

        return {
            "chi_omega": chi_omega,
            "frequencies": omega,
        }

    def real_and_imaginary(self, chi_omega):
        """
        Decompose complex susceptibility into real and imaginary parts.

        Parameters
        ----------
        chi_omega : ndarray (complex)
            Complex susceptibility χ(ω).

        Returns
        -------
        result : dict
            'chi_real': ndarray, χ'(ω) = Re[χ(ω)].
            'chi_imag': ndarray, χ''(ω) = Im[χ(ω)].
            'magnitude': |χ(ω)|.
            'phase': arg(χ(ω)).
        """
        return {
            "chi_real": np.real(chi_omega),
            "chi_imag": np.imag(chi_omega),
            "magnitude": np.abs(chi_omega),
            "phase": np.angle(chi_omega),
        }

    def relaxation_time(self, chi_imag, frequencies):
        """
        Extract relaxation time τ from peak position of χ''(ω).
        The peak of χ'' occurs at ω* = 1/τ for Debye-like relaxation.

        Parameters
        ----------
        chi_imag : ndarray
            Imaginary part χ''(ω).
        frequencies : ndarray
            Angular frequencies ω.

        Returns
        -------
        result : dict
            'tau': relaxation time from peak position.
            'omega_peak': peak frequency.
            'peak_height': χ''(ω*).
            'fwhm': full width at half maximum of χ'' peak.
        """
        # Find peak (use absolute value since χ'' can be negative)
        abs_chi = np.abs(chi_imag)
        peak_idx = np.argmax(abs_chi)
        omega_peak = frequencies[peak_idx]
        peak_height = chi_imag[peak_idx]

        tau = 1.0 / omega_peak if omega_peak > 1e-15 else np.inf

        # Compute FWHM
        half_max = abs_chi[peak_idx] / 2.0
        above_half = abs_chi >= half_max
        indices = np.where(above_half)[0]
        if len(indices) >= 2:
            fwhm = frequencies[indices[-1]] - frequencies[indices[0]]
        else:
            fwhm = np.nan

        return {
            "tau": float(tau),
            "omega_peak": float(omega_peak),
            "peak_height": float(peak_height),
            "fwhm": float(fwhm),
        }

    def cole_cole_fit(self, chi_real, chi_imag, frequencies):
        """
        Fit Cole-Cole model to susceptibility data.
        χ(ω) = χ_∞ + (χ_0 - χ_∞) / (1 + (iωτ)^(1-α))

        Parameters
        ----------
        chi_real : ndarray
            Real part χ'(ω).
        chi_imag : ndarray
            Imaginary part χ''(ω).
        frequencies : ndarray
            Angular frequencies ω (positive).

        Returns
        -------
        result : dict
            'chi_0': static susceptibility.
            'chi_inf': high-frequency susceptibility.
            'tau': relaxation time.
            'alpha': Cole-Cole exponent (0 = Debye).
            'fit_real': fitted χ'(ω).
            'fit_imag': fitted χ''(ω).
            'residual': fit residual norm.
        """

        def cole_cole_model(omega, chi_0, chi_inf, tau, alpha):
            z = (1j * omega * tau) ** (1.0 - alpha)
            chi = chi_inf + (chi_0 - chi_inf) / (1.0 + z)
            return chi

        def residual_func(params):
            chi_0, chi_inf, log_tau, alpha = params
            tau = np.exp(log_tau)
            alpha_c = np.clip(alpha, 0.0, 0.99)
            model = cole_cole_model(frequencies, chi_0, chi_inf, tau, alpha_c)
            res_r = chi_real - np.real(model)
            res_i = chi_imag - np.imag(model)
            return np.concatenate([res_r, res_i])

        # Initial guesses
        chi_0_init = chi_real[0] if len(chi_real) > 0 else 1.0
        chi_inf_init = chi_real[-1] if len(chi_real) > 0 else 0.0
        # Estimate tau from peak of chi_imag
        peak_idx = np.argmax(np.abs(chi_imag))
        omega_peak = frequencies[peak_idx] if frequencies[peak_idx] > 0 else 1.0
        log_tau_init = np.log(1.0 / omega_peak)

        p0 = [chi_0_init, chi_inf_init, log_tau_init, 0.0]

        try:
            result = optimize.least_squares(
                residual_func, p0,
                bounds=([-np.inf, -np.inf, -20, 0.0], [np.inf, np.inf, 20, 0.99]),
                method="trf", max_nfev=5000,
            )
            chi_0, chi_inf, log_tau, alpha = result.x
            tau = np.exp(log_tau)
            residual = np.linalg.norm(result.fun)
        except Exception:
            chi_0, chi_inf, tau, alpha = chi_0_init, chi_inf_init, 1.0 / omega_peak, 0.0
            residual = np.inf

        model_fit = cole_cole_model(frequencies, chi_0, chi_inf, tau, alpha)

        return {
            "chi_0": float(chi_0),
            "chi_inf": float(chi_inf),
            "tau": float(tau),
            "alpha": float(alpha),
            "fit_real": np.real(model_fit),
            "fit_imag": np.imag(model_fit),
            "residual": float(residual),
        }

    def debye_relaxation(self, frequencies, tau, chi_0):
        """
        Compute Debye relaxation susceptibility.
        χ(ω) = χ₀ / (1 + iωτ)

        Parameters
        ----------
        frequencies : ndarray
            Angular frequencies ω.
        tau : float
            Relaxation time.
        chi_0 : float
            Static susceptibility.

        Returns
        -------
        result : dict
            'chi_omega': complex χ(ω).
            'chi_real': χ₀ / (1 + ω²τ²).
            'chi_imag': -χ₀ωτ / (1 + ω²τ²).
        """
        denom = 1.0 + 1j * frequencies * tau
        chi_omega = chi_0 / denom
        return {
            "chi_omega": chi_omega,
            "chi_real": np.real(chi_omega),
            "chi_imag": np.imag(chi_omega),
        }

    def spectral_density(self, chi_imag, frequencies, temperature):
        """
        Compute spectral density from imaginary susceptibility.
        J(ω) = (1 - exp(-βω))⁻¹ · χ''(ω) for bosonic systems.
        At high T: J(ω) ≈ T·χ''(ω)/ω (classical limit).

        Parameters
        ----------
        chi_imag : ndarray
            Imaginary susceptibility χ''(ω).
        frequencies : ndarray
            Angular frequencies ω.
        temperature : float
            Temperature T.

        Returns
        -------
        result : dict
            'J_omega': spectral density J(ω).
            'J_classical': classical spectral density T·χ''(ω)/ω.
            'frequencies': ω values.
        """
        beta = 1.0 / temperature
        J_omega = np.zeros_like(chi_imag)
        J_classical = np.zeros_like(chi_imag)

        for i, w in enumerate(frequencies):
            if abs(w) < 1e-15:
                # Limit ω→0: J(0) = T·χ''(0)/0 → use L'Hopital or set to 0
                J_omega[i] = 0.0
                J_classical[i] = 0.0
            else:
                bw = beta * w
                if abs(bw) < 500:
                    bose = 1.0 / (np.exp(bw) - 1.0) if bw > 0 else -1.0 / (1.0 - np.exp(bw))
                    J_omega[i] = chi_imag[i] * (1.0 + bose) if w > 0 else chi_imag[i] * bose
                else:
                    J_omega[i] = chi_imag[i]
                J_classical[i] = temperature * chi_imag[i] / w

        return {
            "J_omega": J_omega,
            "J_classical": J_classical,
            "frequencies": frequencies,
        }

    def non_linear_susceptibility(self, magnetizations, fields, order=3):
        """
        Compute non-linear susceptibility of given order.
        m(h) = χ₁h + χ₂h² + χ₃h³ + ...
        Extract χ_n by polynomial fitting.

        Parameters
        ----------
        magnetizations : ndarray
            Magnetization values m(h).
        fields : ndarray
            Field values h.
        order : int
            Maximum order to extract (default 3).

        Returns
        -------
        result : dict
            'chi_n': dict mapping order n to χ_n.
            'coefficients': polynomial coefficients.
            'fit_magnetizations': polynomial fit values.
            'residual': fit residual.
        """
        # Fit polynomial m(h) = sum_n chi_n * h^n
        # No constant term if m(h=0) = 0 by symmetry
        n_coeffs = order

        # Build Vandermonde-like matrix without constant term
        A = np.zeros((len(fields), n_coeffs))
        for n in range(n_coeffs):
            A[:, n] = fields ** (n + 1)

        coeffs, residuals, _, _ = np.linalg.lstsq(A, magnetizations, rcond=None)

        chi_n = {}
        for n in range(n_coeffs):
            # Susceptibility of order n+1: m = chi_1 h + chi_2 h^2/2! + ...
            # Actually coefficient of h^(n+1) / (n+1)! gives chi_(n+1)
            chi_n[n + 1] = float(coeffs[n] * np.math.factorial(n + 1))

        fit_m = A @ coeffs
        residual = np.linalg.norm(magnetizations - fit_m)

        return {
            "chi_n": chi_n,
            "coefficients": coeffs,
            "fit_magnetizations": fit_m,
            "residual": float(residual),
        }


class CriticalExponentExtractor:
    """Extract critical exponents from susceptibility and correlation data."""

    def __init__(self, critical_point_estimate=None):
        self.critical_point = critical_point_estimate

    def extract_gamma(self, susceptibility, control_param):
        """
        Extract exponent γ from χ ~ |T - Tc|^{-γ}.

        Parameters
        ----------
        susceptibility : ndarray
            Susceptibility values χ(T).
        control_param : ndarray
            Control parameter values (e.g., temperature).

        Returns
        -------
        result : dict
            'gamma': extracted exponent γ.
            'gamma_error': uncertainty in γ.
            'Tc': critical point used.
            'fit_quality': R² of log-log fit.
        """
        Tc = self._find_critical_point(susceptibility, control_param)
        return self._extract_divergent_exponent(susceptibility, control_param, Tc, "gamma")

    def extract_nu(self, correlation_length, control_param):
        """
        Extract exponent ν from ξ ~ |T - Tc|^{-ν}.

        Parameters
        ----------
        correlation_length : ndarray
            Correlation length values ξ(T).
        control_param : ndarray
            Control parameter values.

        Returns
        -------
        result : dict
            'nu': extracted exponent ν.
            'nu_error': uncertainty in ν.
            'Tc': critical point used.
            'fit_quality': R² of log-log fit.
        """
        Tc = self._find_critical_point(correlation_length, control_param)
        return self._extract_divergent_exponent(correlation_length, control_param, Tc, "nu")

    def extract_eta(self, correlation_at_tc, distances):
        """
        Extract anomalous dimension η from C(r) ~ r^{-(d-2+η)} at Tc.

        Parameters
        ----------
        correlation_at_tc : ndarray
            Correlation function C(r) at T = Tc.
        distances : ndarray
            Distances r.

        Returns
        -------
        result : dict
            'eta': anomalous dimension η.
            'eta_error': uncertainty.
            'effective_dimension': d inferred (assuming d-2+η is the exponent).
            'fit_quality': R² of log-log fit.
        """
        valid = (distances > 0) & (correlation_at_tc > 0)
        log_r = np.log(distances[valid])
        log_C = np.log(correlation_at_tc[valid])

        result = self.log_log_analysis(distances[valid], correlation_at_tc[valid])
        # C(r) ~ r^{-p} where p = d - 2 + η
        p = -result["slope"]  # slope is negative for decaying correlations
        p_err = result["slope_error"]

        return {
            "eta": float(p),  # This is d-2+η; user needs to subtract d-2
            "eta_error": float(p_err),
            "effective_dimension": None,  # Cannot determine d from C(r) alone
            "fit_quality": result["r_squared"],
        }

    def extract_beta(self, order_param, control_param):
        """
        Extract exponent β from m ~ |T - Tc|^β below Tc.

        Parameters
        ----------
        order_param : ndarray
            Order parameter values m(T).
        control_param : ndarray
            Control parameter values.

        Returns
        -------
        result : dict
            'beta': extracted exponent β.
            'beta_error': uncertainty.
            'Tc': critical point used.
            'fit_quality': R² of log-log fit.
        """
        Tc = self._find_critical_point_order_param(order_param, control_param)
        # Select points below Tc where m > 0
        below = control_param < Tc
        if np.sum(below) < 3:
            return {"beta": np.nan, "beta_error": np.nan, "Tc": Tc, "fit_quality": 0.0}

        t_below = control_param[below]
        m_below = order_param[below]
        valid = m_below > 1e-15
        if np.sum(valid) < 3:
            return {"beta": np.nan, "beta_error": np.nan, "Tc": Tc, "fit_quality": 0.0}

        reduced_t = Tc - t_below[valid]
        valid_rt = reduced_t > 1e-15
        if np.sum(valid_rt) < 3:
            return {"beta": np.nan, "beta_error": np.nan, "Tc": Tc, "fit_quality": 0.0}

        result = self.log_log_analysis(reduced_t[valid_rt], m_below[valid][valid_rt])

        return {
            "beta": float(result["slope"]),
            "beta_error": float(result["slope_error"]),
            "Tc": float(Tc),
            "fit_quality": result["r_squared"],
        }

    def scaling_relation_check(self, exponents):
        """
        Verify hyperscaling and scaling relations.

        Checks:
        - Rushbrooke: α + 2β + γ = 2
        - Widom: γ = β(δ - 1)
        - Fisher: γ = ν(2 - η)
        - Josephson (hyperscaling): 2 - α = dν

        Parameters
        ----------
        exponents : dict
            Keys may include 'alpha', 'beta', 'gamma', 'delta', 'nu', 'eta', 'd'.

        Returns
        -------
        result : dict
            'rushbrooke': dict with 'value' (should be 2), 'satisfied' bool.
            'widom': dict if delta available.
            'fisher': dict if nu, eta available.
            'josephson': dict if d, nu, alpha available.
        """
        checks = {}

        alpha = exponents.get("alpha")
        beta = exponents.get("beta")
        gamma = exponents.get("gamma")
        delta = exponents.get("delta")
        nu = exponents.get("nu")
        eta = exponents.get("eta")
        d = exponents.get("d")

        tol = 0.15  # tolerance for scaling relation checks

        if alpha is not None and beta is not None and gamma is not None:
            val = alpha + 2 * beta + gamma
            checks["rushbrooke"] = {
                "value": float(val),
                "expected": 2.0,
                "satisfied": bool(abs(val - 2.0) < tol),
            }

        if gamma is not None and beta is not None and delta is not None:
            val = beta * (delta - 1)
            checks["widom"] = {
                "value": float(val),
                "expected": float(gamma),
                "satisfied": bool(abs(val - gamma) < tol * abs(gamma)),
            }

        if gamma is not None and nu is not None and eta is not None:
            val = nu * (2 - eta)
            checks["fisher"] = {
                "value": float(val),
                "expected": float(gamma),
                "satisfied": bool(abs(val - gamma) < tol * abs(gamma)),
            }

        if alpha is not None and nu is not None and d is not None:
            val = d * nu
            expected = 2 - alpha
            checks["josephson"] = {
                "value": float(val),
                "expected": float(expected),
                "satisfied": bool(abs(val - expected) < tol * abs(expected)),
            }

        return checks

    def fit_power_law(self, x, y, x_critical):
        """
        Fit power law y = A * |x - x_c|^p near the critical point.

        Parameters
        ----------
        x : ndarray
            Independent variable.
        y : ndarray
            Dependent variable.
        x_critical : float
            Critical point x_c.

        Returns
        -------
        result : dict
            'amplitude': prefactor A.
            'exponent': power p.
            'exponent_error': uncertainty in p.
            'fit_y': fitted values.
            'r_squared': goodness of fit.
        """
        delta_x = np.abs(x - x_critical)
        valid = (delta_x > 1e-10) & (y > 0) & np.isfinite(y)
        if np.sum(valid) < 3:
            return {
                "amplitude": np.nan, "exponent": np.nan,
                "exponent_error": np.nan, "fit_y": np.full_like(y, np.nan),
                "r_squared": 0.0,
            }

        log_dx = np.log(delta_x[valid])
        log_y = np.log(np.abs(y[valid]))

        # Weighted linear regression in log-log space
        A_mat = np.vstack([log_dx, np.ones_like(log_dx)]).T
        result = np.linalg.lstsq(A_mat, log_y, rcond=None)
        coeffs = result[0]
        p = coeffs[0]
        log_A = coeffs[1]
        amplitude = np.exp(log_A)

        # Residuals and R²
        y_pred = A_mat @ coeffs
        ss_res = np.sum((log_y - y_pred) ** 2)
        ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
        r_sq = 1.0 - ss_res / max(ss_tot, 1e-15)

        # Error estimate from residuals
        n = len(log_dx)
        if n > 2:
            s2 = ss_res / (n - 2)
            var_p = s2 / np.sum((log_dx - np.mean(log_dx)) ** 2)
            p_err = np.sqrt(max(var_p, 0))
        else:
            p_err = np.nan

        fit_y = np.full_like(y, np.nan, dtype=float)
        fit_y[valid] = amplitude * delta_x[valid] ** p

        return {
            "amplitude": float(amplitude),
            "exponent": float(p),
            "exponent_error": float(p_err),
            "fit_y": fit_y,
            "r_squared": float(r_sq),
        }

    def log_log_analysis(self, x, y):
        """
        Perform log-log analysis with error estimation.

        Parameters
        ----------
        x : ndarray
            Independent variable (positive values).
        y : ndarray
            Dependent variable (positive values).

        Returns
        -------
        result : dict
            'slope': power-law exponent.
            'slope_error': standard error of slope.
            'intercept': log-space intercept.
            'r_squared': R² of log-log fit.
            'log_x': log(x) values used.
            'log_y': log(y) values used.
        """
        valid = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
        if np.sum(valid) < 2:
            return {
                "slope": np.nan, "slope_error": np.nan,
                "intercept": np.nan, "r_squared": 0.0,
                "log_x": np.array([]), "log_y": np.array([]),
            }

        lx = np.log(x[valid])
        ly = np.log(y[valid])

        A = np.vstack([lx, np.ones_like(lx)]).T
        coeffs, residuals, _, _ = np.linalg.lstsq(A, ly, rcond=None)
        slope = coeffs[0]
        intercept = coeffs[1]

        y_pred = A @ coeffs
        ss_res = np.sum((ly - y_pred) ** 2)
        ss_tot = np.sum((ly - np.mean(ly)) ** 2)
        r_sq = 1.0 - ss_res / max(ss_tot, 1e-15)

        n = len(lx)
        if n > 2:
            s2 = ss_res / (n - 2)
            var_slope = s2 / np.sum((lx - np.mean(lx)) ** 2)
            slope_err = np.sqrt(max(var_slope, 0))
        else:
            slope_err = np.nan

        return {
            "slope": float(slope),
            "slope_error": float(slope_err),
            "intercept": float(intercept),
            "r_squared": float(r_sq),
            "log_x": lx,
            "log_y": ly,
        }

    def _find_critical_point(self, observable, control_param):
        """Estimate Tc as location of maximum of observable (divergence)."""
        if self.critical_point is not None:
            return self.critical_point
        idx = np.argmax(np.abs(observable))
        return float(control_param[idx])

    def _find_critical_point_order_param(self, order_param, control_param):
        """Estimate Tc from order parameter vanishing."""
        if self.critical_point is not None:
            return self.critical_point
        # Find where order parameter drops below threshold
        threshold = 0.01 * np.max(np.abs(order_param))
        above = np.abs(order_param) > threshold
        if np.any(above) and np.any(~above):
            # Tc is near the boundary
            transitions = np.diff(above.astype(int))
            idx = np.where(transitions != 0)[0]
            if len(idx) > 0:
                return float(control_param[idx[0]])
        return float(np.median(control_param))

    def _extract_divergent_exponent(self, observable, control_param, Tc, name):
        """Common routine for extracting divergent exponents."""
        delta_t = np.abs(control_param - Tc)
        # Exclude points too close to or at Tc
        cutoff_low = 0.01 * (np.max(control_param) - np.min(control_param))
        cutoff_high = 0.5 * (np.max(control_param) - np.min(control_param))
        valid = (delta_t > cutoff_low) & (delta_t < cutoff_high) & (observable > 0)

        if np.sum(valid) < 3:
            return {name: np.nan, f"{name}_error": np.nan, "Tc": Tc, "fit_quality": 0.0}

        result = self.log_log_analysis(delta_t[valid], observable[valid])
        exponent = -result["slope"]  # Divergent: χ ~ |t|^{-γ} means negative slope

        return {
            name: float(exponent),
            f"{name}_error": float(result["slope_error"]),
            "Tc": float(Tc),
            "fit_quality": result["r_squared"],
        }


class FiniteSizeScaling:
    """Finite-size scaling analysis for phase transitions."""

    def __init__(self, system_sizes, dimension=None):
        self.system_sizes = np.asarray(system_sizes, dtype=float)
        self.dimension = dimension

    def scaling_collapse(self, data_vs_size, control_params, nu_guess, gamma_guess):
        """
        Perform scaling collapse: plot L^{γ/ν} · χ vs L^{1/ν}(T - Tc).

        Parameters
        ----------
        data_vs_size : dict
            Maps system size L to ndarray of observable values (one per control_param).
        control_params : ndarray
            Control parameter values (e.g., temperature).
        nu_guess : float
            Initial guess for correlation length exponent ν.
        gamma_guess : float
            Initial guess for susceptibility exponent γ.

        Returns
        -------
        result : dict
            'nu': optimized ν.
            'gamma': optimized γ.
            'Tc': optimized critical point.
            'collapsed_x': dict of rescaled x-axis per L.
            'collapsed_y': dict of rescaled y-axis per L.
            'quality': quality metric of the collapse.
        """
        sizes = sorted(data_vs_size.keys())

        def collapse_cost(params):
            nu, gamma_over_nu, Tc = params
            if nu <= 0:
                return 1e10
            all_x = []
            all_y = []
            for L in sizes:
                y_data = data_vs_size[L]
                x_scaled = (control_params - Tc) * L ** (1.0 / nu)
                y_scaled = y_data * L ** (-gamma_over_nu)
                all_x.append(x_scaled)
                all_y.append(y_scaled)

            # Measure collapse quality: interpolate each curve and compare
            total_cost = 0.0
            n_pairs = 0
            for i in range(len(sizes)):
                for j in range(i + 1, len(sizes)):
                    # Find overlapping x range
                    x_min = max(np.min(all_x[i]), np.min(all_x[j]))
                    x_max = min(np.max(all_x[i]), np.max(all_x[j]))
                    if x_max <= x_min:
                        continue
                    x_common = np.linspace(x_min, x_max, 50)
                    try:
                        f_i = interpolate.interp1d(
                            all_x[i], all_y[i], kind="linear",
                            bounds_error=False, fill_value=np.nan
                        )
                        f_j = interpolate.interp1d(
                            all_x[j], all_y[j], kind="linear",
                            bounds_error=False, fill_value=np.nan
                        )
                        yi = f_i(x_common)
                        yj = f_j(x_common)
                        mask = np.isfinite(yi) & np.isfinite(yj)
                        if np.sum(mask) > 0:
                            norm = np.mean(np.abs(yi[mask]) + np.abs(yj[mask])) + 1e-15
                            total_cost += np.mean((yi[mask] - yj[mask]) ** 2) / norm ** 2
                            n_pairs += 1
                    except Exception:
                        continue

            return total_cost / max(n_pairs, 1)

        # Initial Tc estimate: midpoint of control params
        Tc_init = np.mean(control_params)
        gamma_nu_init = gamma_guess / nu_guess

        result = optimize.minimize(
            collapse_cost, [nu_guess, gamma_nu_init, Tc_init],
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-8},
        )
        nu_opt, gamma_nu_opt, Tc_opt = result.x
        gamma_opt = gamma_nu_opt * nu_opt

        # Compute collapsed data with optimal parameters
        collapsed_x = {}
        collapsed_y = {}
        for L in sizes:
            collapsed_x[L] = (control_params - Tc_opt) * L ** (1.0 / nu_opt)
            collapsed_y[L] = data_vs_size[L] * L ** (-gamma_nu_opt)

        return {
            "nu": float(nu_opt),
            "gamma": float(gamma_opt),
            "Tc": float(Tc_opt),
            "collapsed_x": collapsed_x,
            "collapsed_y": collapsed_y,
            "quality": float(result.fun),
        }

    def binder_cumulant(self, order_param_samples_vs_size):
        """
        Compute Binder cumulant U₄ = 1 - ⟨m⁴⟩ / (3⟨m²⟩²).

        Parameters
        ----------
        order_param_samples_vs_size : dict
            Maps system size L to ndarray of shape (n_temps, n_samples)
            containing order parameter samples at each temperature.

        Returns
        -------
        result : dict
            Maps system size L to ndarray of U₄ values (one per temperature).
        """
        binder = {}
        for L, samples in order_param_samples_vs_size.items():
            if samples.ndim == 1:
                samples = samples.reshape(1, -1)
            n_temps = samples.shape[0]
            U4 = np.zeros(n_temps)
            for t in range(n_temps):
                s = samples[t, :]
                m2 = np.mean(s ** 2)
                m4 = np.mean(s ** 4)
                if abs(m2) > 1e-15:
                    U4[t] = 1.0 - m4 / (3.0 * m2 ** 2)
                else:
                    U4[t] = 0.0
            binder[L] = U4
        return binder

    def crossing_analysis(self, binder_vs_temp_vs_size):
        """
        Find critical temperature from Binder cumulant crossings.

        Parameters
        ----------
        binder_vs_temp_vs_size : dict
            'temperatures': ndarray of temperature values.
            'binder': dict mapping L to U₄(T) arrays.

        Returns
        -------
        result : dict
            'Tc_estimates': list of Tc from each pair crossing.
            'Tc_mean': mean Tc estimate.
            'Tc_std': standard error.
            'U4_at_Tc': Binder cumulant value at crossing.
            'crossings': list of (L1, L2, Tc) tuples.
        """
        temps = binder_vs_temp_vs_size["temperatures"]
        binder_data = binder_vs_temp_vs_size["binder"]
        sizes = sorted(binder_data.keys())

        crossings = []
        Tc_estimates = []

        for i in range(len(sizes)):
            for j in range(i + 1, len(sizes)):
                L1, L2 = sizes[i], sizes[j]
                U1 = binder_data[L1]
                U2 = binder_data[L2]
                diff = U1 - U2

                # Find zero crossings of the difference
                sign_changes = np.where(np.diff(np.sign(diff)))[0]
                for idx in sign_changes:
                    # Linear interpolation to find crossing temperature
                    t1, t2 = temps[idx], temps[idx + 1]
                    d1, d2 = diff[idx], diff[idx + 1]
                    if abs(d2 - d1) > 1e-15:
                        Tc = t1 - d1 * (t2 - t1) / (d2 - d1)
                        crossings.append((L1, L2, float(Tc)))
                        Tc_estimates.append(float(Tc))

        if len(Tc_estimates) == 0:
            return {
                "Tc_estimates": [], "Tc_mean": np.nan,
                "Tc_std": np.nan, "U4_at_Tc": np.nan, "crossings": [],
            }

        Tc_arr = np.array(Tc_estimates)

        # Estimate U4 at Tc from largest system
        L_max = max(sizes)
        f_binder = interpolate.interp1d(temps, binder_data[L_max], kind="linear",
                                         bounds_error=False, fill_value=np.nan)
        U4_at_Tc = float(f_binder(np.mean(Tc_arr)))

        return {
            "Tc_estimates": Tc_estimates,
            "Tc_mean": float(np.mean(Tc_arr)),
            "Tc_std": float(np.std(Tc_arr) / np.sqrt(len(Tc_arr))),
            "U4_at_Tc": U4_at_Tc,
            "crossings": crossings,
        }

    def finite_size_critical_point(self, observable_vs_size, control_param):
        """
        Extrapolate critical point to infinite size: Tc(L) = Tc(∞) + a·L^{-1/ν}.

        Parameters
        ----------
        observable_vs_size : dict
            Maps system size L to the pseudo-critical point Tc(L).
            (e.g., location of susceptibility peak for each L.)
        control_param : str
            Label for the control parameter (for bookkeeping).

        Returns
        -------
        result : dict
            'Tc_inf': extrapolated Tc(L→∞).
            'Tc_inf_error': uncertainty.
            'one_over_nu': fitted 1/ν exponent.
            'Tc_vs_L': dict of L → Tc(L).
            'fit_quality': R² of the fit.
        """
        sizes = np.array(sorted(observable_vs_size.keys()), dtype=float)
        Tc_L = np.array([observable_vs_size[L] for L in sizes])

        if len(sizes) < 3:
            return {
                "Tc_inf": float(Tc_L[-1]), "Tc_inf_error": np.nan,
                "one_over_nu": np.nan, "Tc_vs_L": observable_vs_size,
                "fit_quality": 0.0,
            }

        # Fit Tc(L) = Tc_inf + a * L^{-omega}
        def model(L, Tc_inf, a, omega):
            return Tc_inf + a * L ** (-omega)

        try:
            # Initial guess: Tc_inf ~ Tc of largest system, omega ~ 1
            p0 = [Tc_L[-1], (Tc_L[0] - Tc_L[-1]) * sizes[0], 1.0]
            popt, pcov = optimize.curve_fit(model, sizes, Tc_L, p0=p0, maxfev=5000)
            Tc_inf, a_coeff, omega = popt
            perr = np.sqrt(np.diag(pcov))

            y_pred = model(sizes, *popt)
            ss_res = np.sum((Tc_L - y_pred) ** 2)
            ss_tot = np.sum((Tc_L - np.mean(Tc_L)) ** 2)
            r_sq = 1.0 - ss_res / max(ss_tot, 1e-15)

            return {
                "Tc_inf": float(Tc_inf),
                "Tc_inf_error": float(perr[0]),
                "one_over_nu": float(omega),
                "Tc_vs_L": observable_vs_size,
                "fit_quality": float(r_sq),
            }
        except (RuntimeError, ValueError):
            # Fallback: linear extrapolation in 1/L
            inv_L = 1.0 / sizes
            coeffs = np.polyfit(inv_L, Tc_L, 1)
            Tc_inf = coeffs[1]  # intercept at 1/L = 0

            y_pred = np.polyval(coeffs, inv_L)
            ss_res = np.sum((Tc_L - y_pred) ** 2)
            ss_tot = np.sum((Tc_L - np.mean(Tc_L)) ** 2)
            r_sq = 1.0 - ss_res / max(ss_tot, 1e-15)

            return {
                "Tc_inf": float(Tc_inf),
                "Tc_inf_error": np.nan,
                "one_over_nu": 1.0,
                "Tc_vs_L": observable_vs_size,
                "fit_quality": float(r_sq),
            }

    def quality_of_collapse(self, collapsed_data):
        """
        Quantify the quality of a scaling collapse using a master curve approach.

        Parameters
        ----------
        collapsed_data : dict
            Maps system size L to dict with 'x' and 'y' arrays of collapsed data.

        Returns
        -------
        result : dict
            'quality': float, quality metric (lower is better, 0 = perfect).
            'pairwise_distances': average distance between curves.
            'master_curve_x': x values of master curve.
            'master_curve_y': y values of master curve.
            'residuals_per_size': dict of per-size residuals from master curve.
        """
        # Collect all collapsed points
        all_x = []
        all_y = []
        all_L = []
        for L, data in collapsed_data.items():
            x = np.asarray(data["x"])
            y = np.asarray(data["y"])
            all_x.append(x)
            all_y.append(y)
            all_L.append(np.full_like(x, L))

        all_x_cat = np.concatenate(all_x)
        all_y_cat = np.concatenate(all_y)

        # Build master curve by binning
        x_min, x_max = np.min(all_x_cat), np.max(all_x_cat)
        n_bins = min(100, len(all_x_cat) // 3)
        if n_bins < 2:
            return {
                "quality": np.nan, "pairwise_distances": np.nan,
                "master_curve_x": np.array([]), "master_curve_y": np.array([]),
                "residuals_per_size": {},
            }

        bin_edges = np.linspace(x_min, x_max, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_means = np.full(n_bins, np.nan)
        bin_vars = np.full(n_bins, np.nan)

        for b in range(n_bins):
            mask = (all_x_cat >= bin_edges[b]) & (all_x_cat < bin_edges[b + 1])
            if np.sum(mask) >= 2:
                bin_means[b] = np.mean(all_y_cat[mask])
                bin_vars[b] = np.var(all_y_cat[mask])

        valid_bins = np.isfinite(bin_means)
        master_x = bin_centers[valid_bins]
        master_y = bin_means[valid_bins]
        variance = bin_vars[valid_bins]

        # Quality = mean variance in bins / mean of master curve squared
        mean_var = np.mean(variance[np.isfinite(variance)])
        mean_y2 = np.mean(master_y ** 2) + 1e-15
        quality = mean_var / mean_y2

        # Per-size residuals
        residuals_per_size = {}
        if len(master_x) >= 2:
            master_interp = interpolate.interp1d(
                master_x, master_y, kind="linear",
                bounds_error=False, fill_value=np.nan
            )
            for i, L in enumerate(collapsed_data.keys()):
                x = np.asarray(collapsed_data[L]["x"])
                y = np.asarray(collapsed_data[L]["y"])
                y_master = master_interp(x)
                mask = np.isfinite(y_master)
                if np.sum(mask) > 0:
                    residuals_per_size[L] = float(np.sqrt(np.mean((y[mask] - y_master[mask]) ** 2)))
                else:
                    residuals_per_size[L] = np.nan

        # Pairwise distances
        sizes = list(collapsed_data.keys())
        pair_dists = []
        for i in range(len(sizes)):
            for j in range(i + 1, len(sizes)):
                xi, yi = np.asarray(collapsed_data[sizes[i]]["x"]), np.asarray(collapsed_data[sizes[i]]["y"])
                xj, yj = np.asarray(collapsed_data[sizes[j]]["x"]), np.asarray(collapsed_data[sizes[j]]["y"])
                x_lo = max(np.min(xi), np.min(xj))
                x_hi = min(np.max(xi), np.max(xj))
                if x_hi <= x_lo:
                    continue
                x_common = np.linspace(x_lo, x_hi, 50)
                try:
                    fi = interpolate.interp1d(xi, yi, bounds_error=False, fill_value=np.nan)
                    fj = interpolate.interp1d(xj, yj, bounds_error=False, fill_value=np.nan)
                    d = fi(x_common) - fj(x_common)
                    mask = np.isfinite(d)
                    if np.sum(mask) > 0:
                        pair_dists.append(np.sqrt(np.mean(d[mask] ** 2)))
                except Exception:
                    continue

        return {
            "quality": float(quality),
            "pairwise_distances": float(np.mean(pair_dists)) if pair_dists else np.nan,
            "master_curve_x": master_x,
            "master_curve_y": master_y,
            "residuals_per_size": residuals_per_size,
        }
