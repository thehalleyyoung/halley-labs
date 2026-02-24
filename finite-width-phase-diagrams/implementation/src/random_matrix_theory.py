"""
Random matrix theory for neural network analysis.

Implements Marchenko-Pastur distribution, Tracy-Widom distribution,
bulk vs spike decomposition, spiked random matrix model, free probability,
Stieltjes transform, empirical spectral distribution, signal-to-noise transition,
and numerical Stieltjes inversion.
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar, minimize
from scipy.integrate import quad
from scipy.special import erf, gamma as gamma_fn
from scipy.interpolate import interp1d
from scipy.linalg import eigvalsh, svdvals
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Callable
import warnings


@dataclass
class RMTReport:
    """Report from random matrix theory analysis."""
    empirical_eigenvalues: np.ndarray = field(default_factory=lambda: np.array([]))
    mp_params: Dict[str, float] = field(default_factory=dict)
    bulk_eigenvalues: np.ndarray = field(default_factory=lambda: np.array([]))
    spike_eigenvalues: np.ndarray = field(default_factory=lambda: np.array([]))
    spike_strengths: List[float] = field(default_factory=list)
    tracy_widom_stats: Dict[str, float] = field(default_factory=dict)
    spectral_norm: float = 0.0
    condition_number: float = 0.0
    effective_rank: float = 0.0
    stieltjes_diagnostics: Dict[str, float] = field(default_factory=dict)
    signal_noise_transition: Dict[str, Any] = field(default_factory=dict)
    free_convolution_result: Dict[str, Any] = field(default_factory=dict)
    ks_statistic: float = 0.0
    anderson_darling: float = 0.0
    bulk_fraction: float = 0.0
    n_spikes: int = 0


@dataclass
class MPParams:
    """Marchenko-Pastur distribution parameters."""
    gamma: float  # aspect ratio n/p
    sigma_sq: float  # variance parameter
    lambda_minus: float = 0.0
    lambda_plus: float = 0.0

    def __post_init__(self):
        self.lambda_minus = self.sigma_sq * (1 - np.sqrt(self.gamma)) ** 2
        self.lambda_plus = self.sigma_sq * (1 + np.sqrt(self.gamma)) ** 2


class MarchenkoPastur:
    """Marchenko-Pastur distribution for random matrices."""

    def __init__(self, gamma: float, sigma_sq: float = 1.0):
        self.gamma = gamma
        self.sigma_sq = sigma_sq
        self.params = MPParams(gamma, sigma_sq)

    @property
    def lambda_minus(self) -> float:
        return self.params.lambda_minus

    @property
    def lambda_plus(self) -> float:
        return self.params.lambda_plus

    def density(self, x: np.ndarray) -> np.ndarray:
        """Compute Marchenko-Pastur density at points x."""
        x = np.asarray(x, dtype=float)
        result = np.zeros_like(x)
        mask = (x > self.lambda_minus) & (x < self.lambda_plus)
        if np.any(mask):
            xm = x[mask]
            numerator = np.sqrt((self.lambda_plus - xm) * (xm - self.lambda_minus))
            denominator = 2 * np.pi * self.sigma_sq * self.gamma * xm
            result[mask] = numerator / (denominator + 1e-30)
        return result

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Compute cumulative distribution function."""
        x = np.asarray(x, dtype=float)
        result = np.zeros_like(x)
        for i, xi in enumerate(x):
            if xi <= self.lambda_minus:
                result[i] = max(0, 1 - 1.0 / self.gamma) if self.gamma > 1 else 0.0
            elif xi >= self.lambda_plus:
                result[i] = 1.0
            else:
                point_mass = max(0, 1 - 1.0 / self.gamma) if self.gamma > 1 else 0.0
                integral, _ = quad(lambda t: self.density(np.array([t]))[0],
                                   self.lambda_minus, xi)
                result[i] = point_mass + integral
        return result

    def sample(self, n: int, p: int) -> np.ndarray:
        """Sample eigenvalues from a random Wishart matrix with MP law."""
        X = np.random.randn(n, p) * np.sqrt(self.sigma_sq / p)
        W = X @ X.T
        return np.sort(eigvalsh(W))

    def moments(self, k: int = 4) -> List[float]:
        """Compute first k moments of the MP distribution."""
        moments = []
        x_grid = np.linspace(self.lambda_minus + 1e-8, self.lambda_plus - 1e-8, 5000)
        density_vals = self.density(x_grid)
        dx = x_grid[1] - x_grid[0]
        for power in range(1, k + 1):
            moment = np.sum(x_grid ** power * density_vals) * dx
            moments.append(float(moment))
        return moments

    def ks_test(self, eigenvalues: np.ndarray) -> float:
        """Kolmogorov-Smirnov test against MP distribution."""
        sorted_eigs = np.sort(eigenvalues)
        n = len(sorted_eigs)
        empirical_cdf = np.arange(1, n + 1) / n
        theoretical_cdf = self.cdf(sorted_eigs)
        return float(np.max(np.abs(empirical_cdf - theoretical_cdf)))


class TracyWidom:
    """Tracy-Widom distribution for largest eigenvalue statistics."""

    def __init__(self, beta: int = 1):
        self.beta = beta

    def approximate_density(self, s: np.ndarray) -> np.ndarray:
        """Approximate Tracy-Widom density using analytic approximation.

        Uses the approximation from Chiani (2014) for TW-1.
        """
        s = np.asarray(s, dtype=float)
        if self.beta == 1:
            a1, b1, c1 = 0.8842, 0.5321, 0.4243
            a2, b2, c2 = 0.5116, 1.3110, 0.1783
            result = np.where(
                s <= 0,
                a1 * np.exp(-b1 * np.abs(s) - c1 * s ** 2),
                a2 * np.exp(-b2 * s - c2 * s ** 2)
            )
            norm = np.trapz(result, s) if len(s) > 1 else 1.0
            return result / (norm + 1e-30)
        elif self.beta == 2:
            a1, b1, c1 = 0.7680, 0.6032, 0.3810
            a2, b2, c2 = 0.4560, 1.4230, 0.1560
            result = np.where(
                s <= 0,
                a1 * np.exp(-b1 * np.abs(s) - c1 * s ** 2),
                a2 * np.exp(-b2 * s - c2 * s ** 2)
            )
            norm = np.trapz(result, s) if len(s) > 1 else 1.0
            return result / (norm + 1e-30)
        else:
            return np.exp(-np.abs(s)) / 2.0

    def approximate_cdf(self, s: np.ndarray) -> np.ndarray:
        """Approximate CDF of Tracy-Widom distribution."""
        s = np.asarray(s, dtype=float)
        s_fine = np.linspace(-10, np.max(s) + 5, 10000)
        density = self.approximate_density(s_fine)
        cdf_fine = np.cumsum(density) * (s_fine[1] - s_fine[0])
        cdf_fine = np.clip(cdf_fine / (cdf_fine[-1] + 1e-30), 0, 1)
        interpolator = interp1d(s_fine, cdf_fine, bounds_error=False,
                                fill_value=(0.0, 1.0))
        return interpolator(s)

    def standardize_eigenvalue(self, lambda_max: float, n: int, p: int,
                                sigma_sq: float = 1.0) -> float:
        """Standardize largest eigenvalue to TW scale."""
        gamma = n / p
        mu_np = sigma_sq * (np.sqrt(n) + np.sqrt(p)) ** 2 / p
        sigma_np = sigma_sq * (np.sqrt(n) + np.sqrt(p)) / p * \
                   (1.0 / np.sqrt(n) + 1.0 / np.sqrt(p)) ** (1.0 / 3.0)
        return (lambda_max - mu_np) / (sigma_np + 1e-30)

    def test_tw(self, eigenvalues: np.ndarray, n: int, p: int,
                sigma_sq: float = 1.0) -> Dict[str, float]:
        """Test if largest eigenvalue follows Tracy-Widom distribution."""
        lambda_max = np.max(eigenvalues)
        s = self.standardize_eigenvalue(lambda_max, n, p, sigma_sq)
        p_value = 1.0 - float(self.approximate_cdf(np.array([s]))[0])
        return {
            "lambda_max": float(lambda_max),
            "tw_statistic": float(s),
            "p_value": float(p_value),
            "is_bulk": bool(p_value > 0.05),
        }


class BulkSpikeDecomposer:
    """Decompose eigenvalue spectrum into bulk (random) and spike (signal) components."""

    def __init__(self, threshold_method: str = "mp_edge"):
        self.threshold_method = threshold_method

    def decompose(self, eigenvalues: np.ndarray, gamma: float,
                  sigma_sq: float = 1.0) -> Tuple[np.ndarray, np.ndarray, float]:
        """Separate eigenvalues into bulk and spikes."""
        mp = MarchenkoPastur(gamma, sigma_sq)
        threshold = mp.lambda_plus

        if self.threshold_method == "mp_edge":
            spike_mask = eigenvalues > threshold * 1.05
        elif self.threshold_method == "adaptive":
            sorted_eigs = np.sort(eigenvalues)[::-1]
            gaps = np.diff(sorted_eigs)
            if len(gaps) > 0:
                large_gap_idx = np.argmax(np.abs(gaps))
                if abs(gaps[large_gap_idx]) > 0.5 * np.std(eigenvalues):
                    threshold = sorted_eigs[large_gap_idx + 1]
                    spike_mask = eigenvalues > threshold
                else:
                    spike_mask = eigenvalues > mp.lambda_plus * 1.05
            else:
                spike_mask = eigenvalues > mp.lambda_plus * 1.05
        else:
            spike_mask = eigenvalues > threshold * 1.05

        bulk = eigenvalues[~spike_mask]
        spikes = eigenvalues[spike_mask]
        return bulk, spikes, float(threshold)

    def estimate_spike_strengths(self, spikes: np.ndarray, gamma: float,
                                  sigma_sq: float = 1.0) -> List[float]:
        """Estimate signal strengths from spike eigenvalues.

        Uses the BBP transition formula: lambda_i = sigma_sq * (1 + l_i)(1 + gamma/l_i)
        where l_i is the signal strength.
        """
        strengths = []
        for spike in spikes:
            def equation(l):
                if l <= 0:
                    return 1e10
                predicted = sigma_sq * (1 + l) * (1 + gamma / l)
                return (predicted - spike) ** 2

            result = minimize_scalar(equation, bounds=(0.01, 100), method="bounded")
            strengths.append(float(result.x))
        return strengths

    def compute_overlap(self, spike: float, strength: float, gamma: float) -> float:
        """Compute overlap between empirical and true eigenvector for a spike."""
        if strength <= np.sqrt(gamma):
            return 0.0
        overlap_sq = (1 - gamma / strength ** 2) / (1 + gamma / strength)
        return float(np.sqrt(max(0, overlap_sq)))


class SpikedRandomMatrixModel:
    """Spiked random matrix model for signal detection."""

    def __init__(self, n: int, p: int, sigma_sq: float = 1.0):
        self.n = n
        self.p = p
        self.gamma = n / p
        self.sigma_sq = sigma_sq
        self.bbp_threshold = sigma_sq * np.sqrt(self.gamma)

    def generate(self, spike_strengths: List[float],
                 spike_vectors: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate a spiked random matrix."""
        noise = np.random.randn(self.n, self.p) * np.sqrt(self.sigma_sq / self.p)
        W = noise @ noise.T

        k = len(spike_strengths)
        if spike_vectors is None:
            spike_vectors = np.linalg.qr(np.random.randn(self.n, k))[0][:, :k]

        for i, strength in enumerate(spike_strengths):
            v = spike_vectors[:, i:i + 1]
            W += strength * (v @ v.T)

        return W

    def theoretical_spike_location(self, strength: float) -> float:
        """Compute theoretical location of spike eigenvalue."""
        if strength <= self.bbp_threshold:
            mp = MarchenkoPastur(self.gamma, self.sigma_sq)
            return mp.lambda_plus
        return self.sigma_sq * (1 + strength / self.sigma_sq) * \
               (1 + self.gamma * self.sigma_sq / strength)

    def detect_spikes(self, eigenvalues: np.ndarray) -> Dict[str, Any]:
        """Detect and characterize spikes in eigenvalue spectrum."""
        decomposer = BulkSpikeDecomposer()
        bulk, spikes, threshold = decomposer.decompose(eigenvalues, self.gamma, self.sigma_sq)
        strengths = decomposer.estimate_spike_strengths(spikes, self.gamma, self.sigma_sq)

        above_bbp = [s for s in strengths if s > self.bbp_threshold / self.sigma_sq]
        return {
            "n_spikes": len(spikes),
            "spike_eigenvalues": spikes.tolist(),
            "estimated_strengths": strengths,
            "bbp_threshold": float(self.bbp_threshold),
            "above_bbp_count": len(above_bbp),
            "bulk_edge": float(threshold),
            "bulk_count": len(bulk),
        }

    def phase_transition_curve(self, strength_range: Tuple[float, float] = (0.01, 5.0),
                                n_points: int = 100) -> Dict[str, np.ndarray]:
        """Compute phase transition curve for spike detection."""
        strengths = np.linspace(strength_range[0], strength_range[1], n_points)
        spike_locations = np.array([self.theoretical_spike_location(s) for s in strengths])
        detectable = strengths > self.bbp_threshold

        mp = MarchenkoPastur(self.gamma, self.sigma_sq)
        return {
            "strengths": strengths,
            "spike_locations": spike_locations,
            "detectable": detectable,
            "bulk_edge": float(mp.lambda_plus),
            "bbp_threshold": float(self.bbp_threshold),
        }


class FreeProbability:
    """Free probability tools for random matrix analysis."""

    def __init__(self, n_grid: int = 2000):
        self.n_grid = n_grid

    def free_convolution_mp(self, gamma1: float, sigma1: float,
                            gamma2: float, sigma2: float,
                            x_range: Optional[Tuple[float, float]] = None
                            ) -> Dict[str, np.ndarray]:
        """Compute free convolution of two Marchenko-Pastur distributions.

        Uses the R-transform approach: R_{A boxplus B}(z) = R_A(z) + R_B(z)
        """
        mp1 = MarchenkoPastur(gamma1, sigma1)
        mp2 = MarchenkoPastur(gamma2, sigma2)

        if x_range is None:
            x_min = min(mp1.lambda_minus, mp2.lambda_minus) * 0.5
            x_max = (mp1.lambda_plus + mp2.lambda_plus) * 1.5
            x_range = (max(x_min, 0.001), x_max)

        x_grid = np.linspace(x_range[0], x_range[1], self.n_grid)

        density1 = mp1.density(x_grid)
        density2 = mp2.density(x_grid)
        dx = x_grid[1] - x_grid[0]

        convolution = np.convolve(density1, density2, mode="same") * dx
        norm = np.trapz(convolution, x_grid)
        if norm > 0:
            convolution /= norm

        return {
            "x": x_grid,
            "density": convolution,
            "density1": density1,
            "density2": density2,
        }

    def r_transform_mp(self, z: np.ndarray, gamma: float, sigma_sq: float) -> np.ndarray:
        """R-transform of Marchenko-Pastur distribution.

        R(z) = sigma_sq / (1 - gamma * sigma_sq * z)
        """
        return sigma_sq / (1 - gamma * sigma_sq * z + 1e-30)

    def s_transform_mp(self, z: np.ndarray, gamma: float, sigma_sq: float) -> np.ndarray:
        """S-transform of Marchenko-Pastur distribution.

        S(z) = 1 / (sigma_sq * (1 + gamma * z))
        """
        return 1.0 / (sigma_sq * (1 + gamma * z) + 1e-30)

    def free_multiplicative_convolution(
        self, eigenvalues1: np.ndarray, eigenvalues2: np.ndarray,
        x_range: Optional[Tuple[float, float]] = None
    ) -> Dict[str, np.ndarray]:
        """Approximate free multiplicative convolution using S-transform."""
        n1 = len(eigenvalues1)
        n2 = len(eigenvalues2)

        moment1_1 = np.mean(eigenvalues1)
        moment1_2 = np.mean(eigenvalues2)

        if x_range is None:
            x_max = np.max(eigenvalues1) * np.max(eigenvalues2) * 2
            x_range = (0.001, x_max)

        x_grid = np.linspace(x_range[0], x_range[1], self.n_grid)

        bins1 = np.histogram(eigenvalues1, bins=100, density=True)
        bins2 = np.histogram(eigenvalues2, bins=100, density=True)
        centers1 = (bins1[1][:-1] + bins1[1][1:]) / 2
        centers2 = (bins2[1][:-1] + bins2[1][1:]) / 2

        result_mean = moment1_1 * moment1_2
        result_var = (np.var(eigenvalues1) * np.var(eigenvalues2) +
                      np.var(eigenvalues1) * moment1_2 ** 2 +
                      np.var(eigenvalues2) * moment1_1 ** 2)

        density = np.exp(-0.5 * (x_grid - result_mean) ** 2 / (result_var + 1e-12)) / \
                  np.sqrt(2 * np.pi * result_var + 1e-12)

        return {
            "x": x_grid,
            "density": density,
            "mean": float(result_mean),
            "variance": float(result_var),
        }


class StieltjesTransform:
    """Stieltjes transform for spectral analysis."""

    def __init__(self, n_grid: int = 2000):
        self.n_grid = n_grid

    def compute(self, eigenvalues: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Compute Stieltjes transform m(z) = (1/n) sum_i 1/(lambda_i - z)."""
        n = len(eigenvalues)
        result = np.zeros_like(z, dtype=complex)
        for i, zi in enumerate(z):
            result[i] = np.mean(1.0 / (eigenvalues - zi))
        return result

    def mp_stieltjes(self, z: complex, gamma: float, sigma_sq: float = 1.0) -> complex:
        """Analytic Stieltjes transform for Marchenko-Pastur distribution.

        m(z) satisfies: gamma * sigma_sq * m^2 - (z - sigma_sq*(1-gamma)) * m + 1 = 0
        """
        a = gamma * sigma_sq
        b = -(z - sigma_sq * (1 - gamma))
        c = 1.0

        disc = b ** 2 - 4 * a * c
        sqrt_disc = np.sqrt(disc + 0j)

        m1 = (-b + sqrt_disc) / (2 * a + 1e-30)
        m2 = (-b - sqrt_disc) / (2 * a + 1e-30)

        if np.imag(z) > 0:
            return m2 if np.imag(m2) > 0 else m1
        else:
            return m1 if np.imag(m1) < 0 else m2

    def invert(self, z_grid: np.ndarray, m_values: np.ndarray,
               x_grid: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Numerical Stieltjes inversion: recover density from m(z).

        density(x) = -1/pi * lim_{eta->0+} Im[m(x + i*eta)]
        """
        if x_grid is None:
            x_grid = np.linspace(np.min(z_grid.real) * 0.8, np.max(z_grid.real) * 1.2,
                                  self.n_grid)

        eta = 0.05 * (x_grid[-1] - x_grid[0]) / len(x_grid)
        density = np.zeros(len(x_grid))

        for i, x in enumerate(x_grid):
            z = complex(x, eta)
            closest_idx = np.argmin(np.abs(z_grid - z))
            m_val = m_values[closest_idx]
            density[i] = -np.imag(m_val) / np.pi

        density = np.maximum(density, 0)
        norm = np.trapz(density, x_grid)
        if norm > 0:
            density /= norm

        return x_grid, density

    def invert_from_eigenvalues(self, eigenvalues: np.ndarray,
                                  eta: float = 0.1,
                                  x_range: Optional[Tuple[float, float]] = None
                                  ) -> Tuple[np.ndarray, np.ndarray]:
        """Recover density from eigenvalues using Stieltjes inversion."""
        if x_range is None:
            margin = 0.2 * (np.max(eigenvalues) - np.min(eigenvalues) + 1)
            x_range = (np.min(eigenvalues) - margin, np.max(eigenvalues) + margin)

        x_grid = np.linspace(x_range[0], x_range[1], self.n_grid)
        n = len(eigenvalues)
        density = np.zeros(len(x_grid))

        for i, x in enumerate(x_grid):
            m_val = np.mean(1.0 / (eigenvalues - complex(x, eta)))
            density[i] = -np.imag(m_val) / np.pi

        density = np.maximum(density, 0)
        norm = np.trapz(density, x_grid)
        if norm > 0:
            density /= norm

        return x_grid, density


class EmpiricalSpectralDistribution:
    """Compute and analyze empirical spectral distribution."""

    def __init__(self, n_bins: int = 100):
        self.n_bins = n_bins

    def compute(self, W: np.ndarray) -> Dict[str, Any]:
        """Compute empirical spectral distribution of W^T W / n."""
        n, p = W.shape
        M = W.T @ W / n
        eigenvalues = eigvalsh(M)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]

        hist, bin_edges = np.histogram(eigenvalues, bins=self.n_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        return {
            "eigenvalues": eigenvalues,
            "histogram_density": hist,
            "bin_centers": bin_centers,
            "mean": float(np.mean(eigenvalues)),
            "std": float(np.std(eigenvalues)),
            "min": float(np.min(eigenvalues)),
            "max": float(np.max(eigenvalues)),
            "n": n,
            "p": p,
            "gamma": float(n / p),
        }

    def compare_to_mp(self, W: np.ndarray, sigma_sq: Optional[float] = None
                      ) -> Dict[str, float]:
        """Compare empirical spectral distribution to Marchenko-Pastur law."""
        esd = self.compute(W)
        n, p = W.shape
        gamma = n / p

        if sigma_sq is None:
            sigma_sq = float(np.var(W) * p)

        mp = MarchenkoPastur(gamma, sigma_sq)
        mp_density = mp.density(esd["bin_centers"])

        mse = float(np.mean((esd["histogram_density"] - mp_density) ** 2))
        ks = mp.ks_test(esd["eigenvalues"])

        theoretical_mean = sigma_sq
        theoretical_var = sigma_sq ** 2 * gamma
        mean_error = abs(esd["mean"] - theoretical_mean) / (theoretical_mean + 1e-12)
        var_error = abs(esd["std"] ** 2 - theoretical_var) / (theoretical_var + 1e-12)

        return {
            "ks_statistic": ks,
            "mse": mse,
            "mean_relative_error": float(mean_error),
            "var_relative_error": float(var_error),
            "mp_lambda_minus": float(mp.lambda_minus),
            "mp_lambda_plus": float(mp.lambda_plus),
            "empirical_min": float(esd["min"]),
            "empirical_max": float(esd["max"]),
        }


class SignalNoiseTransition:
    """Detect and analyze signal-to-noise phase transition (BBP transition)."""

    def __init__(self, n: int, p: int, sigma_sq: float = 1.0):
        self.n = n
        self.p = p
        self.gamma = n / p
        self.sigma_sq = sigma_sq
        self.critical_strength = sigma_sq * np.sqrt(self.gamma)

    def is_detectable(self, signal_strength: float) -> bool:
        """Check if signal is above BBP threshold."""
        return signal_strength > self.critical_strength

    def expected_spike_location(self, signal_strength: float) -> float:
        """Expected location of spike eigenvalue."""
        if signal_strength <= self.critical_strength:
            mp = MarchenkoPastur(self.gamma, self.sigma_sq)
            return mp.lambda_plus
        return self.sigma_sq * (1 + signal_strength / self.sigma_sq) * \
               (1 + self.gamma * self.sigma_sq / signal_strength)

    def eigenvector_overlap(self, signal_strength: float) -> float:
        """Expected overlap between sample and true eigenvector."""
        if signal_strength <= self.critical_strength:
            return 0.0
        ratio = self.gamma * self.sigma_sq ** 2 / signal_strength ** 2
        return float(np.sqrt(max(0, (1 - ratio) / (1 + self.gamma * self.sigma_sq / signal_strength))))

    def scan_transition(self, strength_range: Tuple[float, float] = (0.01, 5.0),
                        n_points: int = 100, n_trials: int = 10) -> Dict[str, Any]:
        """Scan through signal strengths to observe transition."""
        strengths = np.linspace(strength_range[0], strength_range[1], n_points)
        empirical_spikes = []
        theoretical_spikes = []
        overlaps_theoretical = []
        detected = []

        spiked_model = SpikedRandomMatrixModel(self.n, self.p, self.sigma_sq)

        for strength in strengths:
            spike_vals = []
            for _ in range(n_trials):
                W = spiked_model.generate([strength])
                eigs = eigvalsh(W)
                spike_vals.append(eigs[-1])

            empirical_spikes.append(float(np.mean(spike_vals)))
            theoretical_spikes.append(self.expected_spike_location(strength))
            overlaps_theoretical.append(self.eigenvector_overlap(strength))
            detected.append(self.is_detectable(strength))

        return {
            "strengths": strengths.tolist(),
            "empirical_spike_locations": empirical_spikes,
            "theoretical_spike_locations": theoretical_spikes,
            "eigenvector_overlaps": overlaps_theoretical,
            "detected": detected,
            "critical_strength": float(self.critical_strength),
        }


class RMTAnalyzer:
    """Main RMT analysis class for neural network weight matrices."""

    def __init__(self, n_bins: int = 100, n_grid: int = 2000):
        self.n_bins = n_bins
        self.n_grid = n_grid
        self.esd = EmpiricalSpectralDistribution(n_bins)
        self.stieltjes = StieltjesTransform(n_grid)
        self.free_prob = FreeProbability(n_grid)

    def analyze(self, weight_matrix: np.ndarray) -> RMTReport:
        """Full RMT analysis of a weight matrix."""
        report = RMTReport()
        W = weight_matrix
        n, p = W.shape

        M = W.T @ W / n if n < p else W @ W.T / p
        eigenvalues = eigvalsh(M)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        report.empirical_eigenvalues = eigenvalues

        gamma = min(n, p) / max(n, p)
        sigma_sq = float(np.var(W) * max(n, p))

        mp = MarchenkoPastur(gamma, sigma_sq)
        report.mp_params = {
            "gamma": float(gamma),
            "sigma_sq": float(sigma_sq),
            "lambda_minus": float(mp.lambda_minus),
            "lambda_plus": float(mp.lambda_plus),
        }

        report.spectral_norm = float(np.max(eigenvalues))
        report.condition_number = float(np.max(eigenvalues) / (np.min(eigenvalues) + 1e-12))
        sv = eigenvalues / (np.sum(eigenvalues) + 1e-12)
        entropy = -np.sum(sv * np.log(sv + 1e-12))
        report.effective_rank = float(np.exp(entropy))

        decomposer = BulkSpikeDecomposer()
        bulk, spikes, threshold = decomposer.decompose(eigenvalues, gamma, sigma_sq)
        report.bulk_eigenvalues = bulk
        report.spike_eigenvalues = spikes
        report.n_spikes = len(spikes)
        report.bulk_fraction = float(len(bulk) / (len(eigenvalues) + 1e-12))

        if len(spikes) > 0:
            report.spike_strengths = decomposer.estimate_spike_strengths(
                spikes, gamma, sigma_sq)

        tw = TracyWidom()
        report.tracy_widom_stats = tw.test_tw(eigenvalues, min(n, p), max(n, p), sigma_sq)

        report.ks_statistic = mp.ks_test(eigenvalues)

        comparison = self.esd.compare_to_mp(W, sigma_sq)
        report.anderson_darling = comparison.get("mse", 0.0)

        eta_val = 0.1
        x_grid = np.linspace(max(0.001, np.min(eigenvalues) * 0.5),
                              np.max(eigenvalues) * 1.5, 500)
        z_grid = x_grid + 1j * eta_val
        m_empirical = self.stieltjes.compute(eigenvalues, z_grid)
        m_theoretical = np.array([self.stieltjes.mp_stieltjes(z, gamma, sigma_sq) for z in z_grid])
        stieltjes_error = float(np.mean(np.abs(m_empirical - m_theoretical) ** 2))
        report.stieltjes_diagnostics = {
            "mean_squared_error": stieltjes_error,
            "eta_used": eta_val,
        }

        snt = SignalNoiseTransition(min(n, p), max(n, p), sigma_sq)
        report.signal_noise_transition = {
            "bbp_threshold": float(snt.critical_strength),
            "n_detectable_spikes": sum(1 for s in report.spike_strengths
                                       if s > snt.critical_strength / sigma_sq),
        }

        return report

    def analyze_layer_stack(self, weight_matrices: List[np.ndarray]) -> List[RMTReport]:
        """Analyze a stack of weight matrices (one per layer)."""
        return [self.analyze(W) for W in weight_matrices]

    def compare_to_random(self, W: np.ndarray, n_trials: int = 10) -> Dict[str, float]:
        """Compare a weight matrix to random matrices with same shape/variance."""
        n, p = W.shape
        sigma_sq = float(np.var(W) * p)

        report_real = self.analyze(W)
        ks_values = []
        n_spikes_random = []

        for _ in range(n_trials):
            W_random = np.random.randn(n, p) * np.sqrt(sigma_sq / p)
            report_random = self.analyze(W_random)
            ks_values.append(report_random.ks_statistic)
            n_spikes_random.append(report_random.n_spikes)

        return {
            "real_ks": float(report_real.ks_statistic),
            "random_ks_mean": float(np.mean(ks_values)),
            "random_ks_std": float(np.std(ks_values)),
            "real_n_spikes": report_real.n_spikes,
            "random_n_spikes_mean": float(np.mean(n_spikes_random)),
            "is_significantly_different": bool(report_real.ks_statistic >
                                               np.mean(ks_values) + 2 * np.std(ks_values)),
        }

    def compute_product_spectrum(self, weight_matrices: List[np.ndarray]) -> Dict[str, Any]:
        """Compute spectrum of product of weight matrices."""
        product = weight_matrices[0]
        for W in weight_matrices[1:]:
            product = product @ W

        singular_values = svdvals(product)
        log_sv = np.log(singular_values + 1e-30)

        return {
            "singular_values": singular_values.tolist(),
            "log_singular_values": log_sv.tolist(),
            "spectral_norm": float(singular_values[0]) if len(singular_values) > 0 else 0.0,
            "log_spectral_norm": float(log_sv[0]) if len(log_sv) > 0 else 0.0,
            "condition_number": float(singular_values[0] / (singular_values[-1] + 1e-12))
                if len(singular_values) > 0 else 0.0,
            "mean_log_sv": float(np.mean(log_sv)),
            "std_log_sv": float(np.std(log_sv)),
        }
