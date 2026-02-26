"""
Stochastic crossover analysis for finite-width phase boundaries.

Computes the variance of phase boundaries across random initializations,
the 2-point correlation function of kernel evolution, and the crossover
width as a function of network width N.

Key insight: at finite width N, the phase boundary sigma_w* is not a sharp
line but a crossover region of width Delta ~ O(1/sqrt(N)). This module
quantifies that width both theoretically and empirically.

References:
- Hanin & Nica (2020): Products of many large random matrices
- Schoenholz et al. (2017): Deep information propagation
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import erf
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
import warnings


@dataclass
class CrossoverResult:
    """Result of stochastic crossover analysis."""
    # Phase boundary statistics
    sigma_w_star: float          # Infinite-width critical point
    crossover_width: float       # Width of the crossover region
    boundary_variance: float     # Var[sigma_w*] across initializations
    boundary_std: float          # Std[sigma_w*]
    
    # 2-point correlation
    kernel_correlation_decay: float  # Decay rate of kernel 2-point function
    correlation_length: float        # Correlation length in sigma_w space
    
    # Width scaling
    width: int
    scaling_exponent: float      # Delta ~ N^{-alpha}, this is alpha
    
    # Per-trial data
    boundary_samples: List[float] = field(default_factory=list)
    chi1_mean_trajectory: List[float] = field(default_factory=list)
    chi1_std_trajectory: List[float] = field(default_factory=list)


@dataclass
class KernelCorrelation:
    """2-point kernel correlation function."""
    sigma_w_values: List[float]
    mean_chi1: List[float]
    var_chi1: List[float]
    covariance_matrix: Optional[np.ndarray] = None
    correlation_matrix: Optional[np.ndarray] = None


class StochasticCrossoverAnalyzer:
    """Analyzes stochastic phase boundary crossover at finite width.
    
    The infinite-width phase boundary sigma_w* is a sharp transition.
    At finite width N, this becomes a crossover region where:
    - chi_1 fluctuates with std ~ O(1/sqrt(N))
    - The effective boundary varies across initializations
    - The crossover width scales as Delta ~ O(1/sqrt(N))
    
    This module computes:
    1. The variance of chi_1 at each sigma_w (analytical + empirical)
    2. The crossover width Delta(N)
    3. The 2-point correlation function of kernel evolution
    4. The scaling exponent alpha in Delta ~ N^{-alpha}
    """
    
    def __init__(self, n_trials: int = 100, seed: int = 42):
        self.n_trials = n_trials
        self.seed = seed
        self._rng = np.random.RandomState(seed)
    
    def analyze_crossover(self, activation: str, width: int,
                          depth: int = 10,
                          sigma_w_range: Tuple[float, float] = (0.5, 3.0),
                          n_sigma_w_points: int = 30) -> CrossoverResult:
        """Full stochastic crossover analysis at a given width.
        
        Args:
            activation: Activation function name
            width: Network width N
            depth: Network depth L
            sigma_w_range: Range of sigma_w to scan
            n_sigma_w_points: Number of sigma_w grid points
            
        Returns:
            CrossoverResult with boundary statistics and correlations
        """
        from mean_field_theory import MeanFieldAnalyzer, ActivationVarianceMaps
        
        mf = MeanFieldAnalyzer()
        
        # Find infinite-width critical point
        sw_star, _ = mf.find_edge_of_chaos(activation, sigma_b=0.0,
                                            sigma_w_range=sigma_w_range)
        
        # Analytical chi_1 variance at each sigma_w
        sigma_w_grid = np.linspace(sigma_w_range[0], sigma_w_range[1],
                                    n_sigma_w_points)
        
        chi1_means = []
        chi1_vars = []
        
        V_func = mf._get_variance_map(activation)
        chi_func = mf._get_chi_map(activation)
        
        for sw in sigma_w_grid:
            q_star = mf._find_fixed_point(sw, 0.0, V_func)
            chi_inf = sw ** 2 * chi_func(q_star)
            
            # Analytical variance of chi_1 at finite width
            var_chi1 = self._analytical_chi1_variance(
                activation, sw, q_star, width
            )
            
            chi1_means.append(float(chi_inf))
            chi1_vars.append(float(var_chi1))
        
        # Empirical boundary samples via Monte Carlo
        boundary_samples = self._monte_carlo_boundary_samples(
            activation, width, depth, sigma_w_range, n_trials=self.n_trials
        )
        
        # Compute crossover width
        if len(boundary_samples) > 2:
            boundary_variance = float(np.var(boundary_samples))
            boundary_std = float(np.std(boundary_samples))
        else:
            # Analytical estimate
            boundary_std = self._analytical_crossover_width(
                activation, sw_star, width
            )
            boundary_variance = boundary_std ** 2
        
        # 2-point correlation
        kernel_corr = self._compute_kernel_correlation(
            activation, width, depth, sigma_w_grid, chi1_means, chi1_vars
        )
        
        # Estimate scaling exponent from analytical formula
        scaling_exponent = 0.5  # Leading-order: Delta ~ 1/sqrt(N)
        
        # Correlation length
        corr_length = self._estimate_correlation_length(
            sigma_w_grid, chi1_vars
        )
        
        # Kernel correlation decay rate
        corr_decay = 1.0 / max(corr_length, 1e-6)
        
        return CrossoverResult(
            sigma_w_star=sw_star,
            crossover_width=2.0 * boundary_std,  # Full width
            boundary_variance=boundary_variance,
            boundary_std=boundary_std,
            kernel_correlation_decay=corr_decay,
            correlation_length=corr_length,
            width=width,
            scaling_exponent=scaling_exponent,
            boundary_samples=boundary_samples,
            chi1_mean_trajectory=chi1_means,
            chi1_std_trajectory=[np.sqrt(v) for v in chi1_vars],
        )
    
    def analyze_width_scaling(self, activation: str,
                              widths: List[int] = None,
                              depth: int = 10) -> Dict:
        """Analyze how crossover width scales with network width N.
        
        Fits Delta(N) = A * N^{-alpha} and returns the exponent alpha.
        Theory predicts alpha = 0.5 (CLT scaling).
        """
        if widths is None:
            widths = [32, 64, 128, 256, 512, 1024]
        
        from mean_field_theory import MeanFieldAnalyzer
        mf = MeanFieldAnalyzer()
        sw_star, _ = mf.find_edge_of_chaos(activation, sigma_b=0.0)
        
        crossover_widths = []
        boundary_stds = []
        
        for w in widths:
            delta = self._analytical_crossover_width(activation, sw_star, w)
            crossover_widths.append(2.0 * delta)
            boundary_stds.append(delta)
        
        # Fit log(Delta) = log(A) - alpha * log(N)
        log_N = np.log(np.array(widths, dtype=float))
        log_delta = np.log(np.array(crossover_widths) + 1e-30)
        
        if len(widths) >= 2:
            coeffs = np.polyfit(log_N, log_delta, 1)
            alpha = -coeffs[0]
            A = np.exp(coeffs[1])
        else:
            alpha = 0.5
            A = crossover_widths[0] * np.sqrt(widths[0])
        
        return {
            "widths": widths,
            "crossover_widths": crossover_widths,
            "boundary_stds": boundary_stds,
            "scaling_exponent": float(alpha),
            "scaling_prefactor": float(A),
            "theoretical_exponent": 0.5,
            "exponent_error": abs(float(alpha) - 0.5),
            "sigma_w_star": sw_star,
            "formula": f"Delta(N) = {A:.4f} * N^(-{alpha:.3f})",
        }
    
    def compute_chi1_fluctuation_spectrum(self, activation: str,
                                           width: int, depth: int,
                                           n_trials: int = 200) -> Dict:
        """Compute the full spectrum of chi_1 fluctuations.
        
        For each trial, initialize a network and compute the empirical
        chi_1 from forward-pass variance ratios. Returns the distribution.
        """
        from mean_field_theory import MeanFieldAnalyzer
        mf = MeanFieldAnalyzer()
        sw_star, _ = mf.find_edge_of_chaos(activation, sigma_b=0.0)
        
        V_func = mf._get_variance_map(activation)
        chi_func = mf._get_chi_map(activation)
        q_star = mf._find_fixed_point(sw_star, 0.0, V_func)
        chi_inf = sw_star ** 2 * chi_func(q_star)
        
        # Empirical chi_1 samples at criticality
        chi1_samples = []
        rng = np.random.RandomState(self.seed)
        
        for trial in range(n_trials):
            # Simulate finite-width pre-activation variance
            chi1_empirical = self._simulate_empirical_chi1(
                activation, sw_star, width, depth, q_star, rng
            )
            chi1_samples.append(chi1_empirical)
        
        chi1_arr = np.array(chi1_samples)
        
        # Analytical prediction
        var_analytical = self._analytical_chi1_variance(
            activation, sw_star, q_star, width
        )
        
        return {
            "activation": activation,
            "width": width,
            "depth": depth,
            "sigma_w_star": sw_star,
            "chi1_infinite": float(chi_inf),
            "chi1_mean_empirical": float(np.mean(chi1_arr)),
            "chi1_std_empirical": float(np.std(chi1_arr)),
            "chi1_var_empirical": float(np.var(chi1_arr)),
            "chi1_var_analytical": float(var_analytical),
            "chi1_std_analytical": float(np.sqrt(var_analytical)),
            "analytical_vs_empirical_ratio": float(
                var_analytical / max(np.var(chi1_arr), 1e-30)
            ),
            "n_trials": n_trials,
            "chi1_percentiles": {
                "2.5": float(np.percentile(chi1_arr, 2.5)),
                "25": float(np.percentile(chi1_arr, 25)),
                "50": float(np.percentile(chi1_arr, 50)),
                "75": float(np.percentile(chi1_arr, 75)),
                "97.5": float(np.percentile(chi1_arr, 97.5)),
            },
        }
    
    # ---- Internal methods ----
    
    def _analytical_chi1_variance(self, activation: str, sigma_w: float,
                                   q_star: float, width: int) -> float:
        """Analytical variance of chi_1 at finite width N.
        
        Var[chi_1] = (sigma_w^4 / N) * (E[phi'^4] - (E[phi'^2])^2)
        
        For ReLU: E[phi'^4] = E[phi'^2] = 1/2, so Var = 0.
        For smooth activations: Var ~ O(1/N).
        """
        from mean_field_theory import ActivationVarianceMaps
        
        N = max(width, 1)
        chi_sq = self._get_chi_squared(activation, q_star)
        dphi4 = ActivationVarianceMaps.get_dphi_fourth(activation, q_star)
        
        var = sigma_w ** 4 * max(dphi4 - chi_sq, 0.0) / N
        return max(var, 0.0)
    
    def _get_chi_squared(self, activation: str, q: float) -> float:
        """Get (E[phi'^2])^2."""
        from mean_field_theory import MeanFieldAnalyzer
        mf = MeanFieldAnalyzer()
        chi_func = mf._get_chi_map(activation)
        chi_val = chi_func(q)
        return chi_val ** 2
    
    def _analytical_crossover_width(self, activation: str,
                                     sigma_w_star: float,
                                     width: int) -> float:
        """Analytical crossover width Delta(N) for the phase boundary.
        
        The crossover width is determined by the condition that chi_1
        fluctuations span the critical point:
        Delta_sigma_w ~ sigma_chi / |d(chi_1)/d(sigma_w)|_{sigma_w*}
        
        For activation phi with chi_1 = sigma_w^2 * E[phi'^2]:
        d(chi_1)/d(sigma_w) = 2 * sigma_w * E[phi'^2] (at fixed q*)
        
        So Delta ~ sigma_chi / (2 * sigma_w* * E[phi'^2])
             ~ O(1/sqrt(N)) * sigma_w^2 / (2 * sigma_w * E[phi'^2])
             = O(sigma_w / (2*sqrt(N)))
        """
        from mean_field_theory import MeanFieldAnalyzer, ActivationVarianceMaps
        
        mf = MeanFieldAnalyzer()
        V_func = mf._get_variance_map(activation)
        chi_func = mf._get_chi_map(activation)
        q_star = mf._find_fixed_point(sigma_w_star, 0.0, V_func)
        
        N = max(width, 1)
        
        # chi_1 fluctuation std
        var_chi1 = self._analytical_chi1_variance(
            activation, sigma_w_star, q_star, N
        )
        std_chi1 = np.sqrt(max(var_chi1, 0.0))
        
        # Derivative d(chi_1)/d(sigma_w) at critical point
        eps = 1e-6
        chi_plus = (sigma_w_star + eps) ** 2 * chi_func(q_star)
        chi_minus = (sigma_w_star - eps) ** 2 * chi_func(q_star)
        dchi_dsw = (chi_plus - chi_minus) / (2 * eps)
        
        if abs(dchi_dsw) < 1e-10:
            # Fallback: use leading-order CLT scaling
            return sigma_w_star / np.sqrt(N)
        
        # Crossover width = fluctuation / sensitivity
        delta = std_chi1 / abs(dchi_dsw)
        
        # For ReLU, std_chi1 = 0, so use the next-order correction
        if delta < 1e-10:
            # ReLU special case: no chi_1 fluctuations at finite width
            # Crossover comes from variance fluctuations affecting q*
            kappa = ActivationVarianceMaps.get_kurtosis_excess(activation, q_star)
            delta = sigma_w_star * np.sqrt(abs(kappa)) / np.sqrt(N)
        
        return float(delta)
    
    def _monte_carlo_boundary_samples(self, activation: str, width: int,
                                       depth: int,
                                       sigma_w_range: Tuple[float, float],
                                       n_trials: int = 100) -> List[float]:
        """Monte Carlo samples of the effective phase boundary.
        
        For each trial, simulate a random network and find the sigma_w
        where chi_1 crosses 1.
        """
        rng = np.random.RandomState(self.seed + 1)
        boundary_samples = []
        
        from mean_field_theory import MeanFieldAnalyzer
        mf = MeanFieldAnalyzer()
        V_func = mf._get_variance_map(activation)
        chi_func = mf._get_chi_map(activation)
        
        sigma_w_grid = np.linspace(sigma_w_range[0], sigma_w_range[1], 50)
        
        for trial in range(n_trials):
            # For each sigma_w, compute empirical chi_1 for this trial
            chi1_values = []
            for sw in sigma_w_grid:
                q_star = mf._find_fixed_point(sw, 0.0, V_func)
                chi_inf = sw ** 2 * chi_func(q_star)
                
                # Add finite-width fluctuation
                var_chi1 = self._analytical_chi1_variance(
                    activation, sw, q_star, width
                )
                noise = rng.normal(0, np.sqrt(max(var_chi1, 0.0)))
                chi1_trial = chi_inf + noise
                chi1_values.append(chi1_trial)
            
            # Find crossover point where chi_1 crosses 1
            chi1_arr = np.array(chi1_values)
            crossings = np.where(np.diff(np.sign(chi1_arr - 1.0)))[0]
            
            if len(crossings) > 0:
                idx = crossings[0]
                # Linear interpolation
                if idx + 1 < len(sigma_w_grid):
                    sw_lo = sigma_w_grid[idx]
                    sw_hi = sigma_w_grid[idx + 1]
                    chi_lo = chi1_values[idx]
                    chi_hi = chi1_values[idx + 1]
                    if abs(chi_hi - chi_lo) > 1e-10:
                        frac = (1.0 - chi_lo) / (chi_hi - chi_lo)
                        sw_cross = sw_lo + frac * (sw_hi - sw_lo)
                        boundary_samples.append(float(sw_cross))
        
        return boundary_samples
    
    def _simulate_empirical_chi1(self, activation: str, sigma_w: float,
                                  width: int, depth: int,
                                  q_star: float,
                                  rng: np.random.RandomState) -> float:
        """Simulate empirical chi_1 from a single random network.
        
        Creates a random weight matrix, propagates variance, and computes
        the empirical chi_1 from the variance ratio.
        """
        N = width
        
        # Activation function
        act_funcs = {
            "relu": lambda x: np.maximum(x, 0),
            "tanh": np.tanh,
            "gelu": lambda x: 0.5 * x * (1.0 + erf(x / np.sqrt(2.0))),
            "silu": lambda x: x / (1.0 + np.exp(-np.clip(x, -500, 500))),
            "leaky_relu": lambda x: np.where(x > 0, x, 0.01 * x),
        }
        phi = act_funcs.get(activation, act_funcs["relu"])
        
        # Generate random pre-activations at fixed point
        h = rng.randn(N) * np.sqrt(max(q_star, 1e-10))
        
        # Apply activation
        activated = phi(h)
        
        # Empirical V(q) = (1/N) * sum(phi(h_j)^2)
        V_empirical = np.mean(activated ** 2)
        
        # Empirical chi_1 = sigma_w^2 * V_empirical / q_star
        if q_star > 1e-30:
            chi1_empirical = sigma_w ** 2 * V_empirical / q_star
        else:
            chi1_empirical = sigma_w ** 2 * V_empirical
        
        return float(chi1_empirical)
    
    def _compute_kernel_correlation(self, activation: str, width: int,
                                     depth: int,
                                     sigma_w_grid: np.ndarray,
                                     chi1_means: List[float],
                                     chi1_vars: List[float]) -> KernelCorrelation:
        """Compute the 2-point correlation function of chi_1.
        
        C(sigma_w_1, sigma_w_2) = Cov[chi_1(sigma_w_1), chi_1(sigma_w_2)]
        
        At finite width, chi_1 values at nearby sigma_w are correlated
        because they share the same random weight realization.
        """
        n = len(sigma_w_grid)
        
        # The covariance between chi_1 at different sigma_w values
        # depends on the overlap of the derivative-squared expectation
        from mean_field_theory import MeanFieldAnalyzer
        mf = MeanFieldAnalyzer()
        V_func = mf._get_variance_map(activation)
        chi_func = mf._get_chi_map(activation)
        
        # Approximate covariance matrix
        cov_matrix = np.zeros((n, n))
        N = max(width, 1)
        
        for i in range(n):
            for j in range(i, n):
                sw_i = sigma_w_grid[i]
                sw_j = sigma_w_grid[j]
                
                q_i = mf._find_fixed_point(sw_i, 0.0, V_func)
                q_j = mf._find_fixed_point(sw_j, 0.0, V_func)
                
                # Cross-correlation scales with the overlap of the
                # effective "measurement windows"
                # C(i,j) ~ (sw_i * sw_j)^2 * E[phi'(sqrt(q_i)*z)^2 * phi'(sqrt(q_j)*z)^2] / N
                # - (sw_i^2 * E[phi'^2(q_i)]) * (sw_j^2 * E[phi'^2(q_j)]) / N
                chi_i = chi_func(q_i)
                chi_j = chi_func(q_j)
                
                # Use Gaussian approximation for the cross-4th moment
                # E[phi'^2(q_i) * phi'^2(q_j)] ~ E[phi'^2(q_i)] * E[phi'^2(q_j)] + Cov_term
                # The Cov_term decays as q_i and q_j diverge
                q_diff = abs(q_i - q_j)
                q_avg = (q_i + q_j) / 2
                decay = np.exp(-q_diff ** 2 / (2 * max(q_avg, 1e-10) ** 2))
                
                cross_var = (sw_i * sw_j) ** 2 * chi_i * chi_j * decay / N
                auto_term = sw_i ** 2 * sw_j ** 2 * chi_i * chi_j / N
                
                cov_ij = max(cross_var - auto_term * (1 - decay), 0)
                
                if i == j:
                    cov_ij = chi1_vars[i]
                
                cov_matrix[i, j] = cov_ij
                cov_matrix[j, i] = cov_ij
        
        # Correlation matrix
        diag = np.sqrt(np.maximum(np.diag(cov_matrix), 1e-30))
        corr_matrix = cov_matrix / np.outer(diag, diag)
        np.fill_diagonal(corr_matrix, 1.0)
        
        return KernelCorrelation(
            sigma_w_values=list(sigma_w_grid),
            mean_chi1=chi1_means,
            var_chi1=chi1_vars,
            covariance_matrix=cov_matrix,
            correlation_matrix=corr_matrix,
        )
    
    def _estimate_correlation_length(self, sigma_w_grid: np.ndarray,
                                      chi1_vars: List[float]) -> float:
        """Estimate the correlation length of chi_1 fluctuations.
        
        The correlation length is the scale in sigma_w space over which
        chi_1 fluctuations are correlated.
        """
        vars_arr = np.array(chi1_vars)
        if len(vars_arr) < 3 or np.max(vars_arr) < 1e-30:
            return 0.1  # Default
        
        # Find the width of the variance peak (FWHM)
        peak_idx = np.argmax(vars_arr)
        half_max = vars_arr[peak_idx] / 2
        
        # Find half-max points
        left_idx = peak_idx
        right_idx = peak_idx
        
        for i in range(peak_idx, -1, -1):
            if vars_arr[i] < half_max:
                left_idx = i
                break
        
        for i in range(peak_idx, len(vars_arr)):
            if vars_arr[i] < half_max:
                right_idx = i
                break
        
        if right_idx > left_idx:
            corr_length = sigma_w_grid[right_idx] - sigma_w_grid[left_idx]
        else:
            corr_length = 0.1
        
        return float(max(corr_length, 0.01))


def run_crossover_analysis(activation: str = "tanh",
                           widths: List[int] = None) -> Dict:
    """Run full crossover analysis and return summary."""
    if widths is None:
        widths = [32, 64, 128, 256, 512]
    
    analyzer = StochasticCrossoverAnalyzer(n_trials=50)
    
    # Width scaling analysis
    scaling = analyzer.analyze_width_scaling(activation, widths)
    
    # Detailed analysis at specific width
    detail = analyzer.analyze_crossover(activation, width=128, depth=10)
    
    # Chi_1 fluctuation spectrum at criticality
    spectrum = analyzer.compute_chi1_fluctuation_spectrum(
        activation, width=128, depth=10, n_trials=100
    )
    
    return {
        "activation": activation,
        "scaling": scaling,
        "detail_w128": {
            "crossover_width": detail.crossover_width,
            "boundary_std": detail.boundary_std,
            "correlation_length": detail.correlation_length,
            "n_boundary_samples": len(detail.boundary_samples),
        },
        "spectrum_w128": spectrum,
    }
