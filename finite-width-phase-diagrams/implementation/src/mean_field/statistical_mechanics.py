"""
Statistical mechanical treatment of neural networks.

Provides partition functions, entropy computations, thermodynamic potentials,
mean-field thermodynamics, neural-network--specific thermodynamic mappings,
and spin-glass analogies for analyzing neural network phase transitions.
"""

import numpy as np
from scipy import optimize, integrate, linalg, stats
from scipy.special import gammaln, logsumexp
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple, Dict, Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ThermodynamicConfig:
    """Configuration for thermodynamic computations.

    Parameters
    ----------
    width : int
        Network width (number of hidden neurons per layer).
    depth : int
        Network depth (number of hidden layers).
    sigma_w : float
        Weight initialization variance: w ~ N(0, sigma_w^2 / width).
    sigma_b : float
        Bias initialization variance: b ~ N(0, sigma_b^2).
    temperature : float
        Temperature T > 0 controlling the Gibbs measure.
    activation : str
        Activation function identifier ('relu', 'tanh', 'erf').
    n_replicas : int
        Number of replicas for the replica trick.
    """
    width: int = 100
    depth: int = 2
    sigma_w: float = 1.0
    sigma_b: float = 0.0
    temperature: float = 1.0
    activation: str = 'relu'
    n_replicas: int = 0


# ---------------------------------------------------------------------------
# 1. PartitionFunction
# ---------------------------------------------------------------------------

class PartitionFunction:
    r"""Partition function computations for neural network models.

    The partition function Z = \int exp(-S[w]/T) dw governs the
    thermodynamics of the weight-space Gibbs measure.  This class
    provides exact Gaussian results, annealed/quenched approximations,
    saddle-point evaluations, and 1/N series expansions.

    Parameters
    ----------
    config : ThermodynamicConfig
        Thermodynamic configuration.
    """

    def __init__(self, config: ThermodynamicConfig):
        if config.temperature <= 0:
            raise ValueError("Temperature must be positive.")
        self.config = config
        self.T = config.temperature
        self.beta = 1.0 / self.T

    # ----- Gaussian model --------------------------------------------------

    def gaussian_partition(
        self,
        K_ntk: np.ndarray,
        y: np.ndarray,
        noise_var: float,
    ) -> Dict[str, float]:
        r"""Partition function for the Gaussian (kernel regression) model.

        For a Gaussian process model with kernel K and noise variance σ²,
            Z = (2π)^{N/2} |K + σ²I|^{-1/2}
                × exp\bigl(-½ y^T (K + σ²I)^{-1} y\bigr).

        Parameters
        ----------
        K_ntk : np.ndarray, shape (N, N)
            Neural tangent kernel (Gram) matrix.
        y : np.ndarray, shape (N,)
            Training targets.
        noise_var : float
            Observation noise variance σ² > 0.

        Returns
        -------
        dict
            ``log_Z``, ``free_energy``, ``data_fit``, ``complexity`` terms.
        """
        N = K_ntk.shape[0]
        y = np.asarray(y, dtype=np.float64).ravel()
        K_reg = K_ntk + noise_var * np.eye(N)

        # Cholesky for numerical stability
        try:
            L = np.linalg.cholesky(K_reg)
        except np.linalg.LinAlgError:
            # Fall back to eigendecomposition
            eigvals = np.linalg.eigvalsh(K_reg)
            eigvals = np.maximum(eigvals, 1e-14)
            log_det = np.sum(np.log(eigvals))
            alpha = np.linalg.solve(K_reg + 1e-10 * np.eye(N), y)
            data_fit = -0.5 * y @ alpha
            complexity = -0.5 * log_det
            constant = -0.5 * N * np.log(2 * np.pi)
            log_Z = data_fit + complexity + constant
            return {
                'log_Z': float(log_Z),
                'free_energy': float(-self.T * log_Z),
                'data_fit': float(data_fit),
                'complexity': float(complexity),
            }

        alpha = linalg.cho_solve((L, True), y)
        log_det = 2.0 * np.sum(np.log(np.diag(L)))

        data_fit = -0.5 * y @ alpha
        complexity = -0.5 * log_det
        constant = -0.5 * N * np.log(2 * np.pi)
        log_Z = data_fit + complexity + constant

        return {
            'log_Z': float(log_Z),
            'free_energy': float(-self.T * log_Z),
            'data_fit': float(data_fit),
            'complexity': float(complexity),
        }

    # ----- Annealed approximation ------------------------------------------

    def annealed_partition(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int = 1000,
    ) -> Dict[str, float]:
        r"""Annealed approximation: E[Z] via averaging over random weights.

        The annealed free energy F_ann = -T ln E[Z] is an upper bound on
        the quenched free energy F_q = -T E[ln Z].  Here Z is estimated by
        sampling weights from the prior N(0, σ_w²/width).

        Parameters
        ----------
        X : np.ndarray, shape (N, d)
            Input data.
        y : np.ndarray, shape (N,)
            Targets.
        n_samples : int
            Number of Monte Carlo weight samples.

        Returns
        -------
        dict
            ``log_E_Z`` (annealed log-partition), ``F_annealed``.
        """
        rng = np.random.default_rng(42)
        N, d = X.shape
        y = np.asarray(y, dtype=np.float64).ravel()
        width = self.config.width
        sigma_w = self.config.sigma_w

        log_boltzmann_factors = np.empty(n_samples)
        for s in range(n_samples):
            # Single hidden-layer network with random weights
            W = rng.normal(0, sigma_w / np.sqrt(width), size=(d, width))
            a = rng.normal(0, 1.0 / np.sqrt(width), size=(width,))
            hidden = X @ W  # (N, width)

            # Apply activation
            if self.config.activation == 'relu':
                hidden = np.maximum(hidden, 0)
            elif self.config.activation == 'tanh':
                hidden = np.tanh(hidden)
            elif self.config.activation == 'erf':
                from scipy.special import erf
                hidden = erf(hidden)

            predictions = hidden @ a  # (N,)
            loss = 0.5 * np.sum((predictions - y) ** 2)
            log_boltzmann_factors[s] = -self.beta * loss

        # log E[Z] = log (1/S) Σ exp(-βL_s) = logsumexp - log(S)
        log_E_Z = float(logsumexp(log_boltzmann_factors) - np.log(n_samples))
        F_annealed = -self.T * log_E_Z

        return {
            'log_E_Z': log_E_Z,
            'F_annealed': float(F_annealed),
            'n_samples': n_samples,
        }

    # ----- Quenched (replica) approximation --------------------------------

    def quenched_partition(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_replicas: int = 0,
    ) -> Dict[str, float]:
        r"""Quenched free energy via the replica trick.

        Uses E[ln Z] = lim_{n→0} (E[Z^n] - 1)/n.  In practice we compute
        E[Z^n] for several small integer n and extrapolate to n → 0.

        Parameters
        ----------
        X : np.ndarray, shape (N, d)
            Input data.
        y : np.ndarray, shape (N,)
            Targets.
        n_replicas : int
            Maximum replica index n_max for extrapolation.  If 0, uses
            config default.

        Returns
        -------
        dict
            ``E_ln_Z`` (quenched log-partition), ``F_quenched``,
            ``replica_values``.
        """
        if n_replicas <= 0:
            n_replicas = max(self.config.n_replicas, 4)

        rng = np.random.default_rng(123)
        N, d = X.shape
        y = np.asarray(y, dtype=np.float64).ravel()
        n_mc = 500  # Monte Carlo samples for E[Z^n]

        width = self.config.width
        sigma_w = self.config.sigma_w

        def _sample_log_Z():
            """Sample a single log Z from one draw of random data noise."""
            W = rng.normal(0, sigma_w / np.sqrt(width), size=(d, width))
            a = rng.normal(0, 1.0 / np.sqrt(width), size=(width,))
            hidden = X @ W
            if self.config.activation == 'relu':
                hidden = np.maximum(hidden, 0)
            elif self.config.activation == 'tanh':
                hidden = np.tanh(hidden)
            elif self.config.activation == 'erf':
                from scipy.special import erf as _erf
                hidden = _erf(hidden)
            preds = hidden @ a
            loss = 0.5 * np.sum((preds - y) ** 2)
            return -self.beta * loss

        # Collect log Z samples
        log_Z_samples = np.array([_sample_log_Z() for _ in range(n_mc)])

        # E[Z^n] for integer n = 1, ..., n_replicas
        ns = np.arange(1, n_replicas + 1, dtype=float)
        log_E_Zn = np.empty(len(ns))
        for i, n in enumerate(ns):
            log_E_Zn[i] = float(logsumexp(n * log_Z_samples) - np.log(n_mc))

        # Extrapolate (E[Z^n] - 1)/n to n → 0 via polynomial fit
        E_Zn = np.exp(log_E_Zn - log_E_Zn[0])  # normalize for stability
        ratio = (np.exp(log_E_Zn) - 1.0) / ns

        # Linear extrapolation of ratio to n=0
        if len(ns) >= 2:
            coeffs = np.polyfit(ns, ratio, min(2, len(ns) - 1))
            E_ln_Z = float(np.polyval(coeffs, 0.0))
        else:
            E_ln_Z = float(ratio[0])

        F_quenched = -self.T * E_ln_Z

        return {
            'E_ln_Z': E_ln_Z,
            'F_quenched': float(F_quenched),
            'replica_values': {int(n): float(v) for n, v in zip(ns, log_E_Zn)},
        }

    # ----- Saddle-point evaluation -----------------------------------------

    def partition_saddle_point(
        self,
        action_fn: Callable[[np.ndarray], float],
        dim: int,
        x0: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        r"""Saddle-point approximation to the partition function.

        Z ≈ (2π)^{d/2} exp(-S(x*)) / √det(S''(x*))

        where x* is the stationary point of the action S.

        Parameters
        ----------
        action_fn : callable
            Action S(x) to be minimized.
        dim : int
            Dimensionality of the integration domain.
        x0 : np.ndarray or None
            Initial guess for the saddle point.

        Returns
        -------
        dict
            ``x_star``, ``action_value``, ``log_Z``, ``hessian_det``.
        """
        if x0 is None:
            x0 = np.zeros(dim)

        result = optimize.minimize(action_fn, x0, method='L-BFGS-B')
        x_star = result.x
        S_star = result.fun

        # Numerical Hessian via finite differences
        eps = 1e-5
        hess = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(i, dim):
                ei = np.zeros(dim)
                ej = np.zeros(dim)
                ei[i] = eps
                ej[j] = eps
                fpp = action_fn(x_star + ei + ej)
                fpm = action_fn(x_star + ei - ej)
                fmp = action_fn(x_star - ei + ej)
                fmm = action_fn(x_star - ei - ej)
                hess[i, j] = (fpp - fpm - fmp + fmm) / (4 * eps ** 2)
                hess[j, i] = hess[i, j]

        eigvals = np.linalg.eigvalsh(hess)
        eigvals_pos = np.maximum(eigvals, 1e-14)
        log_det = np.sum(np.log(eigvals_pos))

        log_Z = -S_star + 0.5 * dim * np.log(2 * np.pi) - 0.5 * log_det

        return {
            'x_star': x_star,
            'action_value': float(S_star),
            'log_Z': float(log_Z),
            'hessian_det': float(np.exp(log_det)),
            'hessian_eigenvalues': eigvals,
        }

    # ----- Series expansion ------------------------------------------------

    def log_partition_series(
        self,
        X: np.ndarray,
        y: np.ndarray,
        order: int = 4,
    ) -> Dict[str, Any]:
        r"""Expand ln Z as a series in 1/N (width).

        ln Z = N f_0 + f_1 + (1/N) f_2 + O(1/N²)

        where f_0 is the infinite-width (GP) contribution, f_1 is the
        leading finite-width correction, etc.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            Input data.
        y : np.ndarray, shape (n,)
            Targets.
        order : int
            Number of terms in the 1/N expansion.

        Returns
        -------
        dict
            ``coefficients`` list [f_0, f_1, ...], ``log_Z_approx``.
        """
        N_width = self.config.width
        n, d = X.shape
        y = np.asarray(y, dtype=np.float64).ravel()
        sigma_w = self.config.sigma_w

        # f_0: infinite-width GP contribution
        # K_gp = (σ_w² / d) X X^T  for linear activation (leading order)
        K_gp = (sigma_w ** 2 / d) * (X @ X.T)
        gp_result = self.gaussian_partition(K_gp, y, noise_var=self.T)
        f0 = gp_result['log_Z'] / N_width

        # f_1: leading 1/N correction from 4th cumulant of features
        # For ReLU: correction involves kurtosis of pre-activations
        eigvals = np.linalg.eigvalsh(K_gp + self.T * np.eye(n))
        eigvals = np.maximum(eigvals, 1e-14)
        # Trace correction: f_1 ~ -(1/2) Σ ln(1 + correction_i)
        trace_correction = np.sum(1.0 / eigvals ** 2)
        kurtosis_factor = 1.0 if self.config.activation == 'relu' else 0.5
        f1 = -0.5 * kurtosis_factor * trace_correction / N_width

        # Higher-order terms via moment expansion
        coefficients = [float(f0), float(f1)]
        for k in range(2, order):
            # k-th correction scales as N^{-k}
            trace_k = np.sum(1.0 / eigvals ** (k + 1))
            c_k = (-1) ** k * trace_k / (2 * k * N_width ** k)
            coefficients.append(float(c_k))

        # Reconstruct approximate log Z
        log_Z_approx = N_width * coefficients[0]
        for k in range(1, len(coefficients)):
            log_Z_approx += coefficients[k] * N_width ** (1 - k)

        return {
            'coefficients': coefficients,
            'log_Z_approx': float(log_Z_approx),
            'order': order,
        }

    # ----- Temperature dependence ------------------------------------------

    def temperature_dependence(
        self,
        X: np.ndarray,
        y: np.ndarray,
        temperature_range: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        r"""Compute Z(T) and free energy F(T) = -T ln Z over a temperature range.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            Input data.
        y : np.ndarray, shape (n,)
            Targets.
        temperature_range : np.ndarray
            Array of temperatures.

        Returns
        -------
        dict
            ``temperatures``, ``log_Z``, ``free_energy`` arrays.
        """
        temps = np.asarray(temperature_range, dtype=np.float64)
        n, d = X.shape
        sigma_w = self.config.sigma_w

        K_gp = (sigma_w ** 2 / d) * (X @ X.T)

        log_Zs = np.empty(len(temps))
        free_energies = np.empty(len(temps))

        original_T = self.T
        original_beta = self.beta
        for i, T in enumerate(temps):
            if T <= 0:
                log_Zs[i] = -np.inf
                free_energies[i] = np.inf
                continue
            self.T = T
            self.beta = 1.0 / T
            result = self.gaussian_partition(K_gp, y, noise_var=T)
            log_Zs[i] = result['log_Z']
            free_energies[i] = result['free_energy']

        self.T = original_T
        self.beta = original_beta

        return {
            'temperatures': temps,
            'log_Z': log_Zs,
            'free_energy': free_energies,
        }

    # ----- Specific heat ---------------------------------------------------

    def specific_heat(
        self,
        X: np.ndarray,
        y: np.ndarray,
        temperature_range: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        r"""Specific heat C = -T ∂²F/∂T².

        Computed via numerical second derivative of F(T).

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            Input data.
        y : np.ndarray, shape (n,)
            Targets.
        temperature_range : np.ndarray
            Array of temperatures (must have ≥ 3 points for 2nd derivative).

        Returns
        -------
        dict
            ``temperatures``, ``specific_heat``, ``free_energy`` arrays.
        """
        td = self.temperature_dependence(X, y, temperature_range)
        F = td['free_energy']
        temps = td['temperatures']

        # Second derivative via finite differences
        dT = np.diff(temps)
        # Use central differences where possible
        C = np.zeros(len(temps))
        for i in range(1, len(temps) - 1):
            d2F = (F[i + 1] - 2 * F[i] + F[i - 1]) / (0.5 * (dT[i] + dT[i - 1])) ** 2
            C[i] = -temps[i] * d2F
        # Forward/backward at boundaries
        if len(temps) >= 3:
            C[0] = C[1]
            C[-1] = C[-2]

        return {
            'temperatures': temps,
            'specific_heat': C,
            'free_energy': F,
        }

    # ----- Phase transition detection from Z -------------------------------

    def phase_transition_from_partition(
        self,
        X: np.ndarray,
        y: np.ndarray,
        control_param_range: np.ndarray,
    ) -> Dict[str, Any]:
        r"""Detect phase transitions from singularities in Z.

        A phase transition manifests as a non-analyticity in F = -T ln Z.
        First-order transitions show a discontinuity in ∂F/∂α; second-order
        transitions show a divergence in ∂²F/∂α².

        Here α parametrizes temperature: α_i = T_i.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            Input data.
        y : np.ndarray, shape (n,)
            Targets.
        control_param_range : np.ndarray
            Range of control parameter (temperatures).

        Returns
        -------
        dict
            ``transition_points``, ``order``, ``susceptibilities``.
        """
        td = self.temperature_dependence(X, y, control_param_range)
        F = td['free_energy']
        alpha = control_param_range

        # First derivative (energy analog)
        dF = np.gradient(F, alpha)
        # Second derivative (susceptibility analog)
        d2F = np.gradient(dF, alpha)

        # Detect peaks in |d²F/dα²| as transition candidates
        abs_d2F = np.abs(d2F)
        threshold = np.mean(abs_d2F) + 3 * np.std(abs_d2F)

        transition_indices = []
        for i in range(1, len(abs_d2F) - 1):
            if abs_d2F[i] > threshold and abs_d2F[i] > abs_d2F[i - 1] and abs_d2F[i] > abs_d2F[i + 1]:
                transition_indices.append(i)

        transitions = []
        for idx in transition_indices:
            # Classify order: check for discontinuity in dF (first-order)
            # vs divergence in d2F (second-order)
            if idx > 0 and idx < len(dF) - 1:
                dF_jump = abs(dF[idx + 1] - dF[idx - 1])
                if dF_jump > 0.1 * np.std(np.abs(dF)):
                    order = 1
                else:
                    order = 2
            else:
                order = 2
            transitions.append({
                'location': float(alpha[idx]),
                'order': order,
                'susceptibility': float(d2F[idx]),
            })

        return {
            'transition_points': transitions,
            'dF_dalpha': dF,
            'd2F_dalpha2': d2F,
            'free_energy': F,
        }


# ---------------------------------------------------------------------------
# 2. Entropy
# ---------------------------------------------------------------------------

class Entropy:
    r"""Entropy computations for neural network weight spaces.

    Provides microcanonical entropy S(E) = ln Ω(E), configuration entropy
    from partition function data, annealed entropy, function-space entropy,
    weight-space volume estimation, complexity (TAP), and entropy landscapes.

    Parameters
    ----------
    config : ThermodynamicConfig
        Thermodynamic configuration.
    """

    def __init__(self, config: ThermodynamicConfig):
        self.config = config
        self.T = config.temperature
        self.beta = 1.0 / self.T if self.T > 0 else np.inf

    # ----- Microcanonical --------------------------------------------------

    def microcanonical_entropy(
        self,
        energy_level: float,
        K_ntk: np.ndarray,
    ) -> float:
        r"""Microcanonical entropy S(E) = ln Ω(E).

        For the Gaussian model, Ω(E) is the volume of the ellipsoid
        {w : ½ w^T K w ≤ E} in weight space.  Using eigenvalues λ_i of K,

            S(E) = (N/2) ln(2πe E / N) - ½ Σ ln λ_i + ln Γ(N/2)^{-1}

        where N is the dimension of weight space.

        Parameters
        ----------
        energy_level : float
            Energy E at which to compute Ω(E).
        K_ntk : np.ndarray
            Kernel matrix whose eigenvalues define the energy landscape.

        Returns
        -------
        float
            Microcanonical entropy S(E).
        """
        eigvals = np.linalg.eigvalsh(K_ntk)
        eigvals = eigvals[eigvals > 1e-14]
        N = len(eigvals)

        if energy_level <= 0 or N == 0:
            return -np.inf

        # Volume of the ellipsoid with semi-axes √(2E/λ_i)
        log_vol = (N / 2) * np.log(2 * np.pi * np.e * energy_level / N)
        log_vol -= 0.5 * np.sum(np.log(eigvals))
        log_vol -= gammaln(N / 2 + 1)

        return float(log_vol)

    # ----- Configuration entropy -------------------------------------------

    def configuration_entropy(
        self,
        K_ntk: np.ndarray,
        temperature: float,
    ) -> float:
        r"""Configuration entropy S_config = (ln Z + β⟨E⟩) from the Gibbs relation.

        For the Gaussian model with K and noise σ² = T,
            S = ln Z + β ⟨E⟩  =  ½ N (1 + ln 2π) - ½ ln det(K/T + I).

        Parameters
        ----------
        K_ntk : np.ndarray
            Kernel matrix.
        temperature : float
            Temperature.

        Returns
        -------
        float
            Configuration entropy.
        """
        N = K_ntk.shape[0]
        eigvals = np.linalg.eigvalsh(K_ntk)
        eigvals = np.maximum(eigvals, 0)

        # S = ½ Σ [1 + ln(2πT) - ln(λ_i + T)] for each mode
        S = 0.5 * np.sum(1 + np.log(2 * np.pi * temperature)
                         - np.log(eigvals + temperature))
        return float(S)

    # ----- Annealed entropy ------------------------------------------------

    def annealed_entropy(
        self,
        K_ntk: np.ndarray,
        temperature: float,
    ) -> float:
        r"""Annealed entropy S_ann = -∂F_ann/∂T.

        Computed via numerical differentiation of the annealed free energy.

        Parameters
        ----------
        K_ntk : np.ndarray
            Kernel matrix.
        temperature : float
            Temperature at which to evaluate.

        Returns
        -------
        float
            Annealed entropy.
        """
        dT = 1e-4 * temperature
        S_plus = self.configuration_entropy(K_ntk, temperature + dT)
        S_minus = self.configuration_entropy(K_ntk, temperature - dT)

        # F = -T S + U  →  S = -∂F/∂T
        # But configuration_entropy IS S, so dS/dT gives us the
        # temperature derivative; here we return S(T) directly.
        F_plus = -((temperature + dT) * S_plus)
        F_minus = -((temperature - dT) * S_minus)
        S_ann = -(F_plus - F_minus) / (2 * dT)
        return float(S_ann)

    # ----- Function-space entropy ------------------------------------------

    def function_space_entropy(
        self,
        K_ntk: np.ndarray,
        n_functions: int = 1000,
    ) -> float:
        r"""Entropy of the distribution over functions representable by the network.

        Estimates H[f] = -∫ p(f) ln p(f) df via samples from the GP prior
        defined by K_ntk.

        Parameters
        ----------
        K_ntk : np.ndarray, shape (N, N)
            Kernel defining the GP prior on functions.
        n_functions : int
            Number of function samples for entropy estimation.

        Returns
        -------
        float
            Function-space entropy estimate.
        """
        N = K_ntk.shape[0]
        eigvals = np.linalg.eigvalsh(K_ntk)
        eigvals = np.maximum(eigvals, 1e-14)

        # For a Gaussian, H = ½ ln det(2πe K) = ½ [N ln(2πe) + Σ ln λ_i]
        H = 0.5 * (N * np.log(2 * np.pi * np.e) + np.sum(np.log(eigvals)))
        return float(H)

    # ----- Weight-space volume estimation ----------------------------------

    def weight_space_volume(
        self,
        loss_threshold: float,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int = 1000,
    ) -> Dict[str, float]:
        r"""Monte Carlo estimate of weight-space volume Ω = Vol({w : L(w) ≤ E}).

        Samples weights from the prior and counts the fraction with loss ≤ E,
        then multiplies by the prior volume.

        Parameters
        ----------
        loss_threshold : float
            Energy level E.
        X : np.ndarray, shape (n, d)
            Input data.
        y : np.ndarray, shape (n,)
            Targets.
        n_samples : int
            Number of Monte Carlo samples.

        Returns
        -------
        dict
            ``log_volume``, ``fraction``, ``entropy``.
        """
        rng = np.random.default_rng(42)
        n, d = X.shape
        y = np.asarray(y, dtype=np.float64).ravel()
        width = self.config.width
        sigma_w = self.config.sigma_w

        total_params = d * width + width  # weights + readout
        count = 0

        for _ in range(n_samples):
            W = rng.normal(0, sigma_w / np.sqrt(width), size=(d, width))
            a = rng.normal(0, 1.0 / np.sqrt(width), size=(width,))
            hidden = X @ W
            if self.config.activation == 'relu':
                hidden = np.maximum(hidden, 0)
            elif self.config.activation == 'tanh':
                hidden = np.tanh(hidden)
            preds = hidden @ a
            loss = 0.5 * np.sum((preds - y) ** 2)
            if loss <= loss_threshold:
                count += 1

        fraction = count / n_samples
        # Prior volume in a ball of radius ~ σ_w √(total_params)
        R = sigma_w * np.sqrt(total_params)
        log_prior_vol = (total_params / 2) * np.log(2 * np.pi) + total_params * np.log(R)
        log_volume = np.log(max(fraction, 1e-300)) + log_prior_vol
        entropy = log_volume  # S = ln Ω

        return {
            'log_volume': float(log_volume),
            'fraction': float(fraction),
            'entropy': float(entropy),
            'n_samples': n_samples,
        }

    # ----- Complexity function ---------------------------------------------

    def complexity_function(
        self,
        K_ntk: np.ndarray,
        energy: float,
        temperature: float,
    ) -> float:
        r"""Complexity Σ(e) = (1/N) ln N(e): log number of saddles at energy e.

        For a Gaussian random landscape with covariance K, the complexity
        of critical points at energy e is
            Σ(e) = ½ ln N - ½ (e - e_0)² / (2 var) + const
        where e_0 and var come from the spectral density of K.

        Parameters
        ----------
        K_ntk : np.ndarray
            Kernel matrix defining the landscape.
        energy : float
            Energy density e = E/N.
        temperature : float
            Temperature (not directly used for pure complexity, but
            determines thermal weighting).

        Returns
        -------
        float
            Complexity Σ(e).
        """
        N = K_ntk.shape[0]
        eigvals = np.linalg.eigvalsh(K_ntk)
        eigvals = np.maximum(eigvals, 1e-14)

        e0 = np.mean(eigvals)  # typical energy scale
        var_e = np.var(eigvals)

        if var_e < 1e-14:
            return 0.0

        # Gaussian approximation to the density of critical points
        sigma = 0.5 * np.log(N) - 0.5 * (energy - e0) ** 2 / (2 * var_e)
        return float(sigma / N)

    # ----- TAP entropy -----------------------------------------------------

    def tap_entropy(
        self,
        local_magnetizations: np.ndarray,
        couplings: np.ndarray,
    ) -> float:
        r"""Thouless-Anderson-Palmer (TAP) entropy.

        S_TAP = Σ_i s(m_i) - (β²/4) Σ_{ij} J_{ij}² (1 - m_i²)(1 - m_j²)

        where s(m) = -½[(1+m)ln(1+m) + (1-m)ln(1-m)] is the single-site
        entropy and the second term is the Onsager reaction correction.

        Parameters
        ----------
        local_magnetizations : np.ndarray, shape (N,)
            Local magnetizations m_i ∈ [-1, 1].
        couplings : np.ndarray, shape (N, N)
            Coupling matrix J_{ij}.

        Returns
        -------
        float
            TAP entropy.
        """
        m = np.asarray(local_magnetizations, dtype=np.float64).ravel()
        J = np.asarray(couplings, dtype=np.float64)

        # Single-site entropy
        m_clip = np.clip(m, -1 + 1e-14, 1 - 1e-14)
        sp = 1.0 + m_clip
        sm = 1.0 - m_clip
        site_entropy = -0.5 * (sp * np.log(sp) + sm * np.log(sm))
        S_site = np.sum(site_entropy)

        # Onsager correction
        J2 = J ** 2
        one_minus_m2 = 1.0 - m_clip ** 2
        onsager = (self.beta ** 2 / 4) * np.sum(
            J2 * np.outer(one_minus_m2, one_minus_m2)
        )

        return float(S_site - onsager)

    # ----- Entropy landscape -----------------------------------------------

    def entropy_landscape(
        self,
        energy_range: np.ndarray,
        temperature: float,
    ) -> Dict[str, np.ndarray]:
        r"""Compute S(E, T) landscape over an energy range.

        Uses the thermodynamic relation S = (E - F)/T where F is obtained
        from the canonical ensemble at temperature T.

        Parameters
        ----------
        energy_range : np.ndarray
            Array of energy values.
        temperature : float
            Temperature.

        Returns
        -------
        dict
            ``energies``, ``entropy`` arrays.
        """
        energies = np.asarray(energy_range, dtype=np.float64)
        # For a system with N effective degrees of freedom,
        # S(E) ≈ (N/2) ln(E/N) + const  (equipartition)
        N_eff = self.config.width * self.config.depth
        entropies = np.empty_like(energies)
        for i, E in enumerate(energies):
            if E > 0:
                entropies[i] = (N_eff / 2) * np.log(2 * np.pi * np.e * E / N_eff)
            else:
                entropies[i] = -np.inf

        return {
            'energies': energies,
            'entropy': entropies,
            'temperature': temperature,
        }


# ---------------------------------------------------------------------------
# 3. MeanFieldThermodynamics
# ---------------------------------------------------------------------------

class MeanFieldThermodynamics:
    r"""Mean-field thermodynamic computations for neural-network--like models.

    Implements self-consistent equations, Landau expansions, Maxwell
    constructions, and critical-point detection within a mean-field
    framework where the order parameter m satisfies
        m = tanh(β J m + β h).

    Parameters
    ----------
    config : ThermodynamicConfig
        Thermodynamic configuration.
    """

    def __init__(self, config: ThermodynamicConfig):
        if config.temperature <= 0:
            raise ValueError("Temperature must be positive.")
        self.config = config
        self.T = config.temperature
        self.beta = 1.0 / self.T
        # Default coupling strength from weight variance
        self.J = config.sigma_w ** 2

    # ----- Self-consistent equations ---------------------------------------

    def self_consistent_equations(
        self,
        order_params_init: np.ndarray,
        max_iter: int = 1000,
        tol: float = 1e-10,
    ) -> Dict[str, Any]:
        r"""Solve self-consistent mean-field equations iteratively.

        m_i^{new} = tanh(β Σ_j J_{ij} m_j + β h_i)

        Parameters
        ----------
        order_params_init : np.ndarray
            Initial guess for order parameters m.
        max_iter : int
            Maximum iterations.
        tol : float
            Convergence tolerance ||m^{new} - m^{old}||_∞.

        Returns
        -------
        dict
            ``solution``, ``converged``, ``iterations``, ``residual``.
        """
        m = np.asarray(order_params_init, dtype=np.float64).ravel()
        dim = len(m)

        for it in range(max_iter):
            # Mean field: effective field on each site
            h_eff = self.beta * self.J * m  # simplified: uniform coupling
            m_new = np.tanh(h_eff)
            residual = np.max(np.abs(m_new - m))
            m = m_new
            if residual < tol:
                return {
                    'solution': m,
                    'converged': True,
                    'iterations': it + 1,
                    'residual': float(residual),
                }

        return {
            'solution': m,
            'converged': False,
            'iterations': max_iter,
            'residual': float(residual),
        }

    # ----- Magnetization ---------------------------------------------------

    def magnetization(
        self,
        external_field: float,
        temperature: float,
    ) -> float:
        r"""Equilibrium magnetization m = tanh(β(Jm + h)).

        Solves the self-consistency equation for scalar m.

        Parameters
        ----------
        external_field : float
            External field h.
        temperature : float
            Temperature T > 0.

        Returns
        -------
        float
            Equilibrium magnetization m*.
        """
        beta = 1.0 / temperature
        J = self.J

        def _sc_eq(m):
            return m - np.tanh(beta * (J * m + external_field))

        # Solve via Brent's method on [−1, 1] if h > 0, else try multiple starts
        if abs(external_field) > 1e-12:
            try:
                m_star = optimize.brentq(_sc_eq, -1 + 1e-10, 1 - 1e-10)
            except ValueError:
                m_star = optimize.fsolve(_sc_eq, np.sign(external_field) * 0.5)[0]
        else:
            # At h = 0, m = 0 is always a solution; check for non-trivial
            if beta * J > 1:
                # Two stable solutions ±m*
                try:
                    m_star = optimize.brentq(_sc_eq, 0.01, 1 - 1e-10)
                except ValueError:
                    m_star = 0.0
            else:
                m_star = 0.0

        return float(m_star)

    # ----- Internal energy -------------------------------------------------

    def internal_energy(
        self,
        order_params: np.ndarray,
        temperature: float,
    ) -> float:
        r"""Internal energy U = ⟨H⟩ = -J/2 Σ m_i m_j - h Σ m_i.

        For uniform coupling and no external field, U = -J N m² / 2.

        Parameters
        ----------
        order_params : np.ndarray
            Order parameters (magnetizations).
        temperature : float
            Temperature (not directly used for U, but kept for interface).

        Returns
        -------
        float
            Internal energy.
        """
        m = np.asarray(order_params, dtype=np.float64).ravel()
        N = len(m)
        m_mean = np.mean(m)
        U = -0.5 * self.J * N * m_mean ** 2
        return float(U)

    # ----- Helmholtz free energy -------------------------------------------

    def helmholtz_free_energy(
        self,
        order_params: np.ndarray,
        temperature: float,
    ) -> float:
        r"""Helmholtz free energy F = U - TS.

        F = -J/2 N m² + T Σ_i s(m_i)

        where s(m) = ½[(1+m)ln(1+m) + (1−m)ln(1−m)].

        Parameters
        ----------
        order_params : np.ndarray
            Order parameters.
        temperature : float
            Temperature.

        Returns
        -------
        float
            Helmholtz free energy.
        """
        m = np.asarray(order_params, dtype=np.float64).ravel()
        N = len(m)
        m_mean = np.mean(m)

        U = -0.5 * self.J * N * m_mean ** 2

        m_clip = np.clip(m, -1 + 1e-14, 1 - 1e-14)
        sp = 1.0 + m_clip
        sm = 1.0 - m_clip
        entropy_per_site = -0.5 * (sp * np.log(sp) + sm * np.log(sm))
        S = np.sum(entropy_per_site)

        F = U - temperature * S
        return float(F)

    # ----- Gibbs free energy -----------------------------------------------

    def gibbs_free_energy(
        self,
        order_params: np.ndarray,
        temperature: float,
        field: float,
    ) -> float:
        r"""Gibbs free energy G = F - h M where M = N ⟨m⟩.

        Parameters
        ----------
        order_params : np.ndarray
            Order parameters.
        temperature : float
            Temperature.
        field : float
            External field.

        Returns
        -------
        float
            Gibbs free energy.
        """
        F = self.helmholtz_free_energy(order_params, temperature)
        M = np.sum(order_params)
        return float(F - field * M)

    # ----- Susceptibility --------------------------------------------------

    def susceptibility_from_free_energy(
        self,
        temperature_range: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        r"""Susceptibility χ = -∂²F/∂h² = β(1 - m²) in mean field.

        Also computed as χ = ∂m/∂h|_{h=0}.

        Parameters
        ----------
        temperature_range : np.ndarray
            Array of temperatures.

        Returns
        -------
        dict
            ``temperatures``, ``susceptibility`` arrays.
        """
        temps = np.asarray(temperature_range, dtype=np.float64)
        chi = np.empty_like(temps)

        for i, T in enumerate(temps):
            if T <= 0:
                chi[i] = np.inf
                continue
            beta = 1.0 / T
            m = self.magnetization(0.0, T)
            # MF susceptibility: χ = β(1 - m²) / (1 - βJ(1 - m²))
            one_minus_m2 = 1 - m ** 2
            denom = 1 - beta * self.J * one_minus_m2
            if abs(denom) < 1e-14:
                chi[i] = np.inf  # divergence at T_c
            else:
                chi[i] = beta * one_minus_m2 / denom

        return {
            'temperatures': temps,
            'susceptibility': chi,
        }

    # ----- Equation of state -----------------------------------------------

    def equation_of_state(
        self,
        temperature: float,
        field_range: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        r"""Equation of state m(T, h) for the mean-field model.

        Parameters
        ----------
        temperature : float
            Temperature.
        field_range : np.ndarray
            Array of external field values.

        Returns
        -------
        dict
            ``fields``, ``magnetization`` arrays.
        """
        fields = np.asarray(field_range, dtype=np.float64)
        mags = np.array([self.magnetization(h, temperature) for h in fields])
        return {
            'fields': fields,
            'magnetization': mags,
        }

    # ----- Maxwell construction --------------------------------------------

    def maxwell_construction(
        self,
        temperature: float,
        field_range: np.ndarray,
    ) -> Dict[str, Any]:
        r"""Equal-area (Maxwell) construction for first-order transitions.

        At T < T_c with h = 0, the free energy has two minima at ±m*.
        The Maxwell construction finds the field h_c where the two phases
        have equal free energy.

        Parameters
        ----------
        temperature : float
            Temperature below T_c.
        field_range : np.ndarray
            Field range over which to search.

        Returns
        -------
        dict
            ``h_coexistence``, ``m_plus``, ``m_minus``, ``latent_heat``.
        """
        beta = 1.0 / temperature

        if beta * self.J <= 1:
            return {
                'h_coexistence': 0.0,
                'm_plus': 0.0,
                'm_minus': 0.0,
                'latent_heat': 0.0,
                'first_order': False,
            }

        # At h = 0 the two phases have equal F by symmetry
        m_plus = self.magnetization(1e-6, temperature)
        m_minus = -m_plus

        # Latent heat = T ΔS = T [S(m+) - S(m-)]
        # By symmetry ΔS = 0 but the energy jump is ΔE = J(m+² - m-²) = 0
        # For finite h the jump is non-trivial
        latent_heat = abs(
            self.internal_energy(np.array([m_plus]), temperature)
            - self.internal_energy(np.array([m_minus]), temperature)
        )

        return {
            'h_coexistence': 0.0,
            'm_plus': float(m_plus),
            'm_minus': float(m_minus),
            'latent_heat': float(latent_heat),
            'first_order': True,
        }

    # ----- Critical temperature --------------------------------------------

    def critical_temperature(self) -> float:
        r"""Critical temperature T_c = J where the susceptibility diverges.

        In mean-field theory, T_c = J (coupling constant) for the Ising model.

        Returns
        -------
        float
            Critical temperature.
        """
        return float(self.J)

    # ----- Critical field --------------------------------------------------

    def critical_field(self) -> float:
        r"""Critical field h_c for first-order transitions.

        In mean-field Ising, the transition at T < T_c occurs at h = 0.

        Returns
        -------
        float
            Critical field value.
        """
        return 0.0

    # ----- Landau expansion ------------------------------------------------

    def landau_expansion(
        self,
        order_param_range: np.ndarray,
        temperature: float,
    ) -> Dict[str, np.ndarray]:
        r"""Landau free energy F(m) = a_0 + a_2 m² + a_4 m⁴ + a_6 m⁶.

        Computes the exact mean-field F(m) and the Landau polynomial
        approximation.

        Parameters
        ----------
        order_param_range : np.ndarray
            Values of the order parameter m at which to evaluate F.
        temperature : float
            Temperature.

        Returns
        -------
        dict
            ``m_values``, ``F_exact``, ``F_landau``, ``coefficients``.
        """
        ms = np.asarray(order_param_range, dtype=np.float64)

        # Exact MF free energy per site (Ising-like)
        F_exact = np.empty_like(ms)
        for i, m in enumerate(ms):
            F_exact[i] = self.helmholtz_free_energy(
                np.array([m]), temperature
            )

        # Extract Landau coefficients by fitting
        coeffs = self.landau_coefficients(temperature)
        a0, a2, a4, a6 = coeffs['a0'], coeffs['a2'], coeffs['a4'], coeffs['a6']
        F_landau = a0 + a2 * ms ** 2 + a4 * ms ** 4 + a6 * ms ** 6

        return {
            'm_values': ms,
            'F_exact': F_exact,
            'F_landau': F_landau,
            'coefficients': coeffs,
        }

    # ----- Landau coefficients ---------------------------------------------

    def landau_coefficients(
        self,
        temperature: float,
    ) -> Dict[str, float]:
        r"""Extract Landau coefficients a_2(T), a_4(T), a_6(T) from numerical F.

        Expands F(m) around m = 0:
            a_2 = ½ F''(0), a_4 = (1/24) F''''(0), a_6 = (1/720) F''''''(0).

        For the mean-field Ising model:
            a_2 = T/2 - J/2 = ½(T - T_c)
            a_4 = T/12
            a_6 = T/30

        Parameters
        ----------
        temperature : float
            Temperature.

        Returns
        -------
        dict
            ``a0``, ``a2``, ``a4``, ``a6`` Landau coefficients.
        """
        beta = 1.0 / temperature
        J = self.J

        # Analytic MF Landau coefficients
        a0 = 0.0
        a2 = 0.5 * (temperature - J)   # changes sign at T_c = J
        a4 = temperature / 12.0
        a6 = temperature / 30.0

        return {
            'a0': float(a0),
            'a2': float(a2),
            'a4': float(a4),
            'a6': float(a6),
            'T_c': float(J),
        }


# ---------------------------------------------------------------------------
# 4. NeuralNetworkThermodynamics
# ---------------------------------------------------------------------------

class NeuralNetworkThermodynamics:
    r"""Neural-network--specific thermodynamic analysis.

    Maps SGD training concepts to thermodynamic quantities: effective
    temperature from learning rate, Gibbs measure over weights, phase
    coexistence, and the SGD ↔ Langevin correspondence.

    Parameters
    ----------
    config : ThermodynamicConfig
        Thermodynamic configuration.
    """

    def __init__(self, config: ThermodynamicConfig):
        self.config = config
        self.T = config.temperature
        self.beta = 1.0 / self.T if self.T > 0 else np.inf

    # ----- Effective temperature from SGD ----------------------------------

    def training_temperature(
        self,
        lr: float,
        batch_size: int,
        n_data: int,
    ) -> float:
        r"""Effective temperature of SGD dynamics.

        T_eff = η n / (2 B) where η is the learning rate, n is dataset
        size, and B is the batch size.  This follows from the
        fluctuation-dissipation relation for SGD noise.

        Parameters
        ----------
        lr : float
            Learning rate η.
        batch_size : int
            Mini-batch size B.
        n_data : int
            Total number of training samples.

        Returns
        -------
        float
            Effective temperature T_eff.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return float(lr * n_data / (2 * batch_size))

    # ----- Gibbs measure ---------------------------------------------------

    def gibbs_measure(
        self,
        weights: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        temperature: float,
    ) -> float:
        r"""Log-probability under the Gibbs measure P(w) ∝ exp(-L(w)/T).

        Parameters
        ----------
        weights : np.ndarray
            Network weight vector (flattened).
        X : np.ndarray, shape (n, d)
            Input data.
        y : np.ndarray, shape (n,)
            Targets.
        temperature : float
            Temperature T > 0.

        Returns
        -------
        float
            Log-probability ln P(w) up to normalization constant.
        """
        d = X.shape[1]
        width = self.config.width

        # Unpack weights into first-layer W and readout a
        n_first = d * width
        if len(weights) < n_first + width:
            raise ValueError("Weight vector too short for specified architecture.")
        W = weights[:n_first].reshape(d, width)
        a = weights[n_first:n_first + width]

        hidden = X @ W
        if self.config.activation == 'relu':
            hidden = np.maximum(hidden, 0)
        elif self.config.activation == 'tanh':
            hidden = np.tanh(hidden)
        elif self.config.activation == 'erf':
            from scipy.special import erf
            hidden = erf(hidden)

        preds = hidden @ a
        loss = 0.5 * np.sum((preds - y) ** 2)

        return float(-loss / temperature)

    # ----- Free energy landscape -------------------------------------------

    def free_energy_landscape(
        self,
        X: np.ndarray,
        y: np.ndarray,
        temperature: float,
        param_range: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        r"""Free energy as a function of a summary statistic (weight norm).

        Computes F(||w||) = -T ln ∫ δ(||w|| - r) exp(-L(w)/T) dw
        via saddle-point approximation.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            Input data.
        y : np.ndarray, shape (n,)
            Targets.
        temperature : float
            Temperature.
        param_range : np.ndarray
            Values of the summary statistic (weight norm r).

        Returns
        -------
        dict
            ``param_values``, ``free_energy`` arrays.
        """
        n, d = X.shape
        width = self.config.width
        total_params = d * width + width
        rs = np.asarray(param_range, dtype=np.float64)
        F = np.empty_like(rs)

        rng = np.random.default_rng(42)
        n_mc = 200

        for i, r in enumerate(rs):
            if r <= 0:
                F[i] = np.inf
                continue

            log_boltz = []
            for _ in range(n_mc):
                # Sample weights on the sphere of radius r
                w = rng.normal(size=total_params)
                w = r * w / np.linalg.norm(w)
                W = w[:d * width].reshape(d, width)
                a = w[d * width:d * width + width]
                hidden = X @ W
                if self.config.activation == 'relu':
                    hidden = np.maximum(hidden, 0)
                elif self.config.activation == 'tanh':
                    hidden = np.tanh(hidden)
                preds = hidden @ a
                loss = 0.5 * np.sum((preds - y) ** 2)
                log_boltz.append(-loss / temperature)

            log_boltz = np.array(log_boltz)
            # F(r) ≈ -T [logsumexp(log_boltz) - ln(n_mc)]
            # Plus entropic contribution from sphere volume
            log_sphere = (total_params - 1) * np.log(r) + (
                total_params / 2) * np.log(2 * np.pi) - gammaln(total_params / 2)
            log_Z_r = logsumexp(log_boltz) - np.log(n_mc) + log_sphere
            F[i] = -temperature * log_Z_r

        return {
            'param_values': rs,
            'free_energy': F,
        }

    # ----- Phase coexistence -----------------------------------------------

    def phase_coexistence(
        self,
        X: np.ndarray,
        y: np.ndarray,
        temperature_range: np.ndarray,
    ) -> Dict[str, Any]:
        r"""Find coexistence temperature where two phases have equal free energy.

        Looks for the temperature T_coex where the free energy landscape
        F(r; T) has two equal minima.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            Input data.
        y : np.ndarray, shape (n,)
            Targets.
        temperature_range : np.ndarray
            Temperature range to search.

        Returns
        -------
        dict
            ``T_coexistence``, ``r_phase1``, ``r_phase2``.
        """
        # Simplified: check for bimodality in the weight norm distribution
        # at different temperatures
        best_T = None
        min_gap = np.inf

        for T in temperature_range:
            r_range = np.linspace(0.1, 5.0, 30)
            landscape = self.free_energy_landscape(X, y, T, r_range)
            F = landscape['free_energy']

            # Find local minima
            minima = []
            for j in range(1, len(F) - 1):
                if F[j] < F[j - 1] and F[j] < F[j + 1]:
                    minima.append((r_range[j], F[j]))

            if len(minima) >= 2:
                gap = abs(minima[0][1] - minima[1][1])
                if gap < min_gap:
                    min_gap = gap
                    best_T = T
                    r1, r2 = minima[0][0], minima[1][0]

        if best_T is not None:
            return {
                'T_coexistence': float(best_T),
                'r_phase1': float(r1),
                'r_phase2': float(r2),
                'free_energy_gap': float(min_gap),
            }
        return {
            'T_coexistence': None,
            'r_phase1': None,
            'r_phase2': None,
            'free_energy_gap': None,
        }

    # ----- Latent heat -----------------------------------------------------

    def latent_heat(
        self,
        X: np.ndarray,
        y: np.ndarray,
        temperature_c: float,
    ) -> float:
        r"""Latent heat ΔE at a first-order transition at temperature T_c.

        L = T_c ΔS where ΔS is the entropy discontinuity.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        y : np.ndarray
            Targets.
        temperature_c : float
            Coexistence temperature.

        Returns
        -------
        float
            Latent heat.
        """
        dT = 0.01 * temperature_c
        pf = PartitionFunction(self.config)

        # Energy just below and above T_c
        K = (self.config.sigma_w ** 2 / X.shape[1]) * (X @ X.T)

        self_below = pf.gaussian_partition(K, y, noise_var=temperature_c - dT)
        self_above = pf.gaussian_partition(K, y, noise_var=temperature_c + dT)

        # E = -∂ ln Z / ∂β
        E_below = -(self_below['log_Z']) * (temperature_c - dT)
        E_above = -(self_above['log_Z']) * (temperature_c + dT)

        return float(abs(E_above - E_below))

    # ----- Correlation function --------------------------------------------

    def correlation_function(
        self,
        weights_samples: np.ndarray,
        i: int,
        j: int,
    ) -> float:
        r"""Two-point correlation ⟨w_i w_j⟩ - ⟨w_i⟩⟨w_j⟩.

        Parameters
        ----------
        weights_samples : np.ndarray, shape (n_samples, n_params)
            Weight samples from the Gibbs measure.
        i, j : int
            Parameter indices.

        Returns
        -------
        float
            Connected correlation function C(i, j).
        """
        wi = weights_samples[:, i]
        wj = weights_samples[:, j]
        return float(np.mean(wi * wj) - np.mean(wi) * np.mean(wj))

    # ----- Correlation length ----------------------------------------------

    def correlation_length(
        self,
        correlation_fn: np.ndarray,
        distance_range: np.ndarray,
    ) -> float:
        r"""Extract correlation length ξ from exponential decay C(r) ~ exp(-r/ξ).

        Fits ln|C(r)| = const - r/ξ to extract ξ.

        Parameters
        ----------
        correlation_fn : np.ndarray
            Correlation function values C(r).
        distance_range : np.ndarray
            Distance values r.

        Returns
        -------
        float
            Correlation length ξ.
        """
        C = np.asarray(correlation_fn, dtype=np.float64)
        r = np.asarray(distance_range, dtype=np.float64)

        # Filter out zero/negative correlations
        mask = C > 1e-14
        if np.sum(mask) < 2:
            return np.inf

        log_C = np.log(C[mask])
        r_filt = r[mask]

        # Linear fit: ln C = a - r/ξ
        coeffs = np.polyfit(r_filt, log_C, 1)
        slope = coeffs[0]

        if slope >= 0:
            return np.inf  # no decay
        xi = -1.0 / slope
        return float(xi)

    # ----- Order parameter distribution ------------------------------------

    def order_parameter_distribution(
        self,
        X: np.ndarray,
        y: np.ndarray,
        temperature: float,
        n_samples: int = 500,
    ) -> Dict[str, np.ndarray]:
        r"""Distribution P(q) of the overlap order parameter.

        Samples weight configurations from the Gibbs measure and computes
        the pairwise overlap q = (1/N) w_a · w_b.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            Input data.
        y : np.ndarray, shape (n,)
            Targets.
        temperature : float
            Temperature.
        n_samples : int
            Number of weight samples.

        Returns
        -------
        dict
            ``q_values``, ``P_q`` (histogram).
        """
        rng = np.random.default_rng(42)
        d = X.shape[1]
        width = self.config.width
        sigma_w = self.config.sigma_w
        n_params = d * width + width

        # Collect weight samples (MCMC-like: importance sampling from prior)
        weights_list = []
        log_probs = []
        for _ in range(n_samples):
            W = rng.normal(0, sigma_w / np.sqrt(width), size=(d, width))
            a = rng.normal(0, 1.0 / np.sqrt(width), size=(width,))
            w = np.concatenate([W.ravel(), a])

            hidden = X @ W
            if self.config.activation == 'relu':
                hidden = np.maximum(hidden, 0)
            elif self.config.activation == 'tanh':
                hidden = np.tanh(hidden)
            preds = hidden @ a
            loss = 0.5 * np.sum((preds - y) ** 2)

            weights_list.append(w)
            log_probs.append(-loss / temperature)

        weights_arr = np.array(weights_list)
        log_probs = np.array(log_probs)
        # Importance weights
        log_w = log_probs - logsumexp(log_probs)
        imp_weights = np.exp(log_w)

        # Compute pairwise overlaps
        overlaps = []
        for a_idx in range(min(n_samples, 100)):
            for b_idx in range(a_idx + 1, min(n_samples, 100)):
                q = np.dot(weights_arr[a_idx], weights_arr[b_idx]) / n_params
                overlaps.append(q)

        overlaps = np.array(overlaps) if overlaps else np.array([0.0])

        # Histogram
        hist, bin_edges = np.histogram(overlaps, bins=50, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        return {
            'q_values': bin_centers,
            'P_q': hist,
            'overlaps_raw': overlaps,
        }

    # ----- (T, N) phase diagram --------------------------------------------

    def compute_phase_diagram_temperature_width(
        self,
        T_range: np.ndarray,
        width_range: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        r"""Compute (T, N) phase diagram.

        Classifies each (T, N) point as lazy/rich/critical by comparing
        the NTK at initialization with the infinite-width limit.

        Parameters
        ----------
        T_range : np.ndarray
            Temperature values.
        width_range : np.ndarray
            Width values.
        X : np.ndarray
            Input data.
        y : np.ndarray
            Targets.

        Returns
        -------
        dict
            ``T_grid``, ``N_grid``, ``phase_labels`` (2D array).
        """
        T_arr = np.asarray(T_range, dtype=np.float64)
        N_arr = np.asarray(width_range, dtype=np.int64)
        phase_labels = np.empty((len(T_arr), len(N_arr)), dtype=int)

        for i, T in enumerate(T_arr):
            for j, N in enumerate(N_arr):
                beta = 1.0 / T
                # Rough classification:
                # - Large N, low T → lazy (phase 0)
                # - Small N, high T → rich (phase 1)
                # - Intermediate → critical (phase 2)
                sigma_w = self.config.sigma_w
                control = beta * sigma_w ** 2 * N
                if control > 10:
                    phase_labels[i, j] = 0  # lazy
                elif control < 0.1:
                    phase_labels[i, j] = 1  # rich
                else:
                    phase_labels[i, j] = 2  # critical

        return {
            'T_grid': T_arr,
            'N_grid': N_arr,
            'phase_labels': phase_labels,
        }

    # ----- Loss as energy --------------------------------------------------

    def train_loss_as_energy(
        self,
        loss_trajectory: np.ndarray,
    ) -> Dict[str, Any]:
        r"""Interpret the training loss trajectory as energy in a thermodynamic framework.

        Computes running temperature estimates T(t) = (2/N) E(t) (equipartition)
        and detects thermal equilibrium.

        Parameters
        ----------
        loss_trajectory : np.ndarray
            Loss values over training steps.

        Returns
        -------
        dict
            ``energy``, ``temperature_estimate``, ``equilibrated`` flag,
            ``equilibration_time``.
        """
        L = np.asarray(loss_trajectory, dtype=np.float64)
        N_eff = self.config.width

        T_estimate = 2.0 * L / N_eff

        # Detect equilibration: when running variance stabilizes
        window = max(len(L) // 10, 10)
        running_var = np.array([
            np.var(L[max(0, k - window):k + 1]) for k in range(len(L))
        ])

        # Equilibrated when var change < 10% for sustained period
        equilibrated = False
        eq_time = len(L)
        for k in range(window, len(running_var)):
            recent = running_var[k - window:k]
            if len(recent) > 1 and np.std(recent) < 0.1 * np.mean(recent):
                equilibrated = True
                eq_time = k - window
                break

        return {
            'energy': L,
            'temperature_estimate': T_estimate,
            'equilibrated': equilibrated,
            'equilibration_time': int(eq_time),
        }

    # ----- SGD as Langevin -------------------------------------------------

    def sgd_as_langevin(
        self,
        lr: float,
        gradient_noise_cov: np.ndarray,
    ) -> Dict[str, Any]:
        r"""Map SGD to Langevin dynamics: dw = -∇L dt + √(2T) dW.

        The correspondence gives:
            T_eff = (η/2) tr(Σ) / dim(w)
            diffusion matrix D = (η/2) Σ

        where Σ is the gradient noise covariance and η is the learning rate.

        Parameters
        ----------
        lr : float
            Learning rate η.
        gradient_noise_cov : np.ndarray, shape (p, p)
            Gradient noise covariance matrix Σ.

        Returns
        -------
        dict
            ``T_effective``, ``diffusion_matrix``, ``anisotropy_ratio``.
        """
        Sigma = np.asarray(gradient_noise_cov, dtype=np.float64)
        dim = Sigma.shape[0]

        D = (lr / 2) * Sigma
        T_eff = np.trace(D) / dim

        eigvals = np.linalg.eigvalsh(Sigma)
        eigvals = np.maximum(eigvals, 1e-14)
        anisotropy = eigvals[-1] / eigvals[0]

        return {
            'T_effective': float(T_eff),
            'diffusion_matrix': D,
            'anisotropy_ratio': float(anisotropy),
            'eigenvalues': eigvals,
        }


# ---------------------------------------------------------------------------
# 5. SpinGlassAnalogy
# ---------------------------------------------------------------------------

class SpinGlassAnalogy:
    r"""Spin-glass models and analogies for neural networks.

    Maps neural network problems to disordered spin systems, providing
    tools for computing replica-symmetric free energies, the de Almeida–
    Thouless instability line, overlap distributions, TAP complexity,
    Gardner volumes, and the SAT/UNSAT transition.

    Parameters
    ----------
    config : ThermodynamicConfig
        Thermodynamic configuration.
    """

    def __init__(self, config: ThermodynamicConfig):
        self.config = config
        self.T = config.temperature
        self.beta = 1.0 / self.T if self.T > 0 else np.inf

    # ----- Sherrington-Kirkpatrick free energy ------------------------------

    def sherrington_kirkpatrick_free_energy(
        self,
        J_matrix: np.ndarray,
        temperature: float,
    ) -> Dict[str, float]:
        r"""Sherrington-Kirkpatrick model free energy.

        F_SK = -½ β Σ_{ij} J_{ij}² q² - T ln 2 - T ⟨ln cosh(β√(Σ J² q) z)⟩_z

        in the replica-symmetric approximation, where q is the overlap
        order parameter and z is a standard Gaussian random variable.

        Parameters
        ----------
        J_matrix : np.ndarray, shape (N, N)
            Coupling matrix.
        temperature : float
            Temperature.

        Returns
        -------
        dict
            ``free_energy``, ``overlap_q``, ``entropy``, ``internal_energy``.
        """
        N = J_matrix.shape[0]
        beta = 1.0 / temperature
        J2 = np.mean(J_matrix ** 2)

        # Solve for q self-consistently:  q = <tanh²(β√(J² q) z)>_z
        def _sc_eq(q):
            if q < 0:
                return q
            h_scale = beta * np.sqrt(J2 * max(q, 1e-14))
            # Numerical average over z ~ N(0,1)
            z_pts = np.linspace(-5, 5, 200)
            w_pts = stats.norm.pdf(z_pts)
            tanh2 = np.tanh(h_scale * z_pts) ** 2
            return q - np.trapz(tanh2 * w_pts, z_pts)

        # Find fixed point
        try:
            q_star = optimize.brentq(_sc_eq, 0, 1 - 1e-10)
        except ValueError:
            q_star = 0.0

        # RS free energy per spin
        h_scale = beta * np.sqrt(J2 * max(q_star, 1e-14))
        z_pts = np.linspace(-5, 5, 200)
        w_pts = stats.norm.pdf(z_pts)
        log_cosh_avg = np.trapz(
            np.log(np.cosh(h_scale * z_pts)) * w_pts, z_pts
        )

        f = (-0.5 * beta * J2 * (1 - q_star) ** 2
             - temperature * np.log(2)
             - temperature * log_cosh_avg)

        # Entropy S = -∂F/∂T (numerical)
        dT = 1e-4 * temperature
        # Quick re-evaluation at T ± dT
        f_list = []
        for T_shift in [temperature - dT, temperature + dT]:
            b = 1.0 / T_shift
            h_s = b * np.sqrt(J2 * max(q_star, 1e-14))
            lc = np.trapz(
                np.log(np.cosh(h_s * z_pts)) * w_pts, z_pts
            )
            f_list.append(
                -0.5 * b * J2 * (1 - q_star) ** 2
                - T_shift * np.log(2)
                - T_shift * lc
            )
        S = -(f_list[1] - f_list[0]) / (2 * dT)
        U = f + temperature * S

        return {
            'free_energy': float(f),
            'overlap_q': float(q_star),
            'entropy': float(S),
            'internal_energy': float(U),
        }

    # ----- NN as spin glass ------------------------------------------------

    def nn_as_spin_glass(
        self,
        K_ntk: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        r"""Map a neural network to a spin glass with couplings from the NTK.

        The NTK kernel K defines effective couplings J_{ij} ∝ K_{ij}
        between data points (viewed as "spins" in function space).

        Parameters
        ----------
        K_ntk : np.ndarray, shape (N, N)
            NTK kernel matrix.
        X : np.ndarray
            Input data.
        y : np.ndarray
            Targets.

        Returns
        -------
        dict
            ``J_effective`` (coupling matrix), ``h_effective`` (fields),
            ``glass_temperature``.
        """
        N = K_ntk.shape[0]
        y = np.asarray(y, dtype=np.float64).ravel()

        # Effective couplings: normalize K to have unit diagonal on average
        diag_mean = np.mean(np.diag(K_ntk))
        if diag_mean > 1e-14:
            J_eff = K_ntk / (diag_mean * N)
        else:
            J_eff = K_ntk / N

        # Effective fields from the targets
        h_eff = J_eff @ y

        # Glass temperature = √(Var(J_{ij}))
        off_diag = J_eff[np.triu_indices(N, k=1)]
        T_glass = float(np.std(off_diag)) if len(off_diag) > 0 else 0.0

        return {
            'J_effective': J_eff,
            'h_effective': h_eff,
            'glass_temperature': T_glass,
            'N_spins': N,
        }

    # ----- Replica-symmetric free energy -----------------------------------

    def replica_symmetric_free_energy(
        self,
        q: float,
        temperature: float,
    ) -> float:
        r"""Replica-symmetric free energy f(q) for the SK model.

        f_RS(q) = -β J² (1-q)² / 4 - T ln 2
                  - T ⟨ln cosh(β J √q z)⟩_z
                  + β J² q² / 4

        Parameters
        ----------
        q : float
            Overlap order parameter q ∈ [0, 1].
        temperature : float
            Temperature.

        Returns
        -------
        float
            RS free energy density.
        """
        beta = 1.0 / temperature
        J = self.config.sigma_w

        h_scale = beta * J * np.sqrt(max(q, 1e-14))
        z_pts = np.linspace(-6, 6, 300)
        w_pts = stats.norm.pdf(z_pts)
        log_cosh = np.trapz(
            np.log(np.cosh(h_scale * z_pts)) * w_pts, z_pts
        )

        f = (-0.25 * beta * J ** 2 * (1 - q) ** 2
             - temperature * np.log(2)
             - temperature * log_cosh
             + 0.25 * beta * J ** 2 * q ** 2)

        return float(f)

    # ----- de Almeida–Thouless line ----------------------------------------

    def de_almeida_thouless_line(
        self,
        temperature_range: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        r"""Compute the de Almeida–Thouless (AT) line for RSB onset.

        The RS solution becomes unstable when
            β²J² ⟨(1 - tanh²(βJ√q z))²⟩_z = 1.

        Parameters
        ----------
        temperature_range : np.ndarray
            Temperature values.

        Returns
        -------
        dict
            ``temperatures``, ``h_AT`` (AT field), ``stable`` (RS stability flag).
        """
        temps = np.asarray(temperature_range, dtype=np.float64)
        J = self.config.sigma_w
        stable = np.ones(len(temps), dtype=bool)
        h_AT = np.zeros(len(temps))

        z_pts = np.linspace(-6, 6, 300)
        w_pts = stats.norm.pdf(z_pts)

        for i, T in enumerate(temps):
            if T <= 0:
                stable[i] = False
                continue

            beta = 1.0 / T

            # Find RS overlap q at h = 0
            def _sc(q):
                if q < 0:
                    return q
                h_s = beta * J * np.sqrt(max(q, 1e-14))
                tanh2 = np.tanh(h_s * z_pts) ** 2
                return q - np.trapz(tanh2 * w_pts, z_pts)

            try:
                q_star = optimize.brentq(_sc, 0, 1 - 1e-10)
            except ValueError:
                q_star = 0.0

            # AT stability criterion
            h_s = beta * J * np.sqrt(max(q_star, 1e-14))
            sech4 = (1.0 - np.tanh(h_s * z_pts) ** 2) ** 2
            lhs = beta ** 2 * J ** 2 * np.trapz(sech4 * w_pts, z_pts)

            stable[i] = lhs < 1.0
            # AT field: h where lhs = 1
            h_AT[i] = 0.0  # at h = 0 for the standard SK model

        return {
            'temperatures': temps,
            'h_AT': h_AT,
            'stable': stable,
        }

    # ----- Overlap distribution --------------------------------------------

    def overlap_distribution(
        self,
        weight_samples: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        r"""Compute P(q) from multiple weight-space solutions.

        The overlap between replicas a, b is
            q_{ab} = (1/N) Σ_i w_i^a w_i^b.

        Parameters
        ----------
        weight_samples : np.ndarray, shape (n_replicas, n_params)
            Weight vectors from different solutions/replicas.

        Returns
        -------
        dict
            ``q_values``, ``P_q`` (histogram).
        """
        n_replicas, n_params = weight_samples.shape
        overlaps = []

        for a in range(n_replicas):
            for b in range(a + 1, n_replicas):
                q = np.dot(weight_samples[a], weight_samples[b]) / n_params
                overlaps.append(q)

        overlaps = np.array(overlaps) if overlaps else np.array([0.0])
        hist, bin_edges = np.histogram(overlaps, bins=50, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        return {
            'q_values': bin_centers,
            'P_q': hist,
            'mean_q': float(np.mean(overlaps)),
            'std_q': float(np.std(overlaps)),
        }

    # ----- Complexity (TAP) ------------------------------------------------

    def complexity(
        self,
        energy: float,
        temperature: float,
    ) -> float:
        r"""TAP complexity Σ(e, T) = (1/N) ln N(e, T).

        In the SK model the complexity is
            Σ(e) = ½ ln 2 - ½ (e/J)² + ...

        for energies near the ground state.

        Parameters
        ----------
        energy : float
            Energy density e = E/N.
        temperature : float
            Temperature.

        Returns
        -------
        float
            Complexity.
        """
        J = self.config.sigma_w
        if J < 1e-14:
            return 0.0

        # Gaussian approximation
        sigma_e = J / np.sqrt(2)
        sigma_val = 0.5 * np.log(2) - 0.5 * (energy / J) ** 2

        # Threshold energy below which complexity is negative (no states)
        if sigma_val < 0:
            return 0.0
        return float(sigma_val)

    # ----- Gardner volume --------------------------------------------------

    def gardner_volume(
        self,
        K_ntk: np.ndarray,
        y: np.ndarray,
        kappa_range: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        r"""Gardner volume for margin κ.

        V(κ) = Vol({w : y_i (K w)_i ≥ κ for all i}) measures the
        fraction of weight space achieving margin ≥ κ.

        For a Gaussian weight distribution, this reduces to a product of
        error functions involving the eigenvalues of K.

        Parameters
        ----------
        K_ntk : np.ndarray, shape (N, N)
            Kernel matrix.
        y : np.ndarray, shape (N,)
            Binary labels ±1.
        kappa_range : np.ndarray
            Margin values.

        Returns
        -------
        dict
            ``kappa``, ``log_volume``, ``entropy``.
        """
        from scipy.special import erfc

        N = K_ntk.shape[0]
        y = np.asarray(y, dtype=np.float64).ravel()
        kappas = np.asarray(kappa_range, dtype=np.float64)

        eigvals, eigvecs = np.linalg.eigh(K_ntk)
        eigvals = np.maximum(eigvals, 1e-14)

        # Project targets onto eigenbasis
        y_proj = eigvecs.T @ y

        log_vols = np.empty(len(kappas))
        for i, kappa in enumerate(kappas):
            # For each eigenmode, probability of satisfying margin
            log_prob = 0.0
            for k in range(N):
                # Effective SNR for mode k
                snr = abs(y_proj[k]) * np.sqrt(eigvals[k])
                # P(margin ≥ κ) ≈ ½ erfc((κ - snr) / √2)
                arg = (kappa - snr) / np.sqrt(2)
                log_p = np.log(max(0.5 * erfc(arg), 1e-300))
                log_prob += log_p
            log_vols[i] = log_prob

        return {
            'kappa': kappas,
            'log_volume': log_vols,
            'entropy': log_vols / N,
        }

    # ----- SAT/UNSAT transition --------------------------------------------

    def sat_unsat_transition(
        self,
        K_ntk: np.ndarray,
        alpha_range: np.ndarray,
    ) -> Dict[str, Any]:
        r"""Satisfiability transition for random labels at load α = N/P.

        At α < α_c, random labels can be fitted (SAT); at α > α_c they
        cannot (UNSAT).  For the perceptron, α_c = 2 (Cover's theorem,
        large-P limit).

        Parameters
        ----------
        K_ntk : np.ndarray, shape (N, N)
            Kernel matrix.
        alpha_range : np.ndarray
            Load parameter α = n_samples / n_params.

        Returns
        -------
        dict
            ``alpha``, ``sat_probability``, ``alpha_c``.
        """
        rng = np.random.default_rng(42)
        N = K_ntk.shape[0]
        eigvals = np.linalg.eigvalsh(K_ntk)
        eigvals = np.maximum(eigvals, 1e-14)

        alphas = np.asarray(alpha_range, dtype=np.float64)
        sat_prob = np.empty(len(alphas))
        n_trials = 50

        for i, alpha in enumerate(alphas):
            n_effective = max(int(alpha * N), 1)
            n_sat = 0
            for _ in range(n_trials):
                y_rand = rng.choice([-1, 1], size=min(n_effective, N))
                K_sub = K_ntk[:len(y_rand), :len(y_rand)]
                # Check if system is satisfiable: K w = y has a solution
                # with margin > 0 iff min eigenvalue of (y diag) K (y diag) > 0
                try:
                    w = np.linalg.solve(
                        K_sub + 1e-8 * np.eye(len(y_rand)), y_rand
                    )
                    margin = np.min(y_rand * (K_sub @ w))
                    if margin > -1e-6:
                        n_sat += 1
                except np.linalg.LinAlgError:
                    pass
            sat_prob[i] = n_sat / n_trials

        # Find α_c where sat_prob crosses 0.5
        alpha_c = None
        for i in range(len(sat_prob) - 1):
            if sat_prob[i] >= 0.5 > sat_prob[i + 1]:
                # Linear interpolation
                frac = (0.5 - sat_prob[i + 1]) / (sat_prob[i] - sat_prob[i + 1])
                alpha_c = alphas[i + 1] - frac * (alphas[i + 1] - alphas[i])
                break

        return {
            'alpha': alphas,
            'sat_probability': sat_prob,
            'alpha_c': float(alpha_c) if alpha_c is not None else None,
        }
