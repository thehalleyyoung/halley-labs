"""
Fourier Accountant for DP-Forge.

Implements the characteristic function (Fourier transform) based privacy
accountant from Koskela et al. (2020). Provides FFT-based composition of
privacy guarantees via characteristic functions with numerical stability
improvements and heterogeneous mechanism support.

Key Features:
    - Characteristic function computation for mechanisms
    - FFT-based composition via CF multiplication
    - Conversion from CF to (ε, δ) via numerical inversion
    - Heterogeneous composition (different mechanisms per step)
    - Log-CF representation for numerical stability
    - Comparison with RDP-based bounds

References:
    - Koskela, A., Jälkö, J., & Honkela, A. (2020). Computing tight
      differential privacy guarantees using FFT. In AISTATS 2020.
    - Doroshenko, V., et al. (2022). Connect the dots: Tighter discrete
      approximations of privacy loss distributions.

Classes:
    FourierAccountant — FFT-based privacy accountant via CFs

Functions:
    characteristic_function — Compute CF of privacy loss distribution
    compose_cf              — Compose CFs via multiplication
    cf_to_epsilon           — Convert CF to (ε, δ) guarantee
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from numpy.fft import fft, ifft, fftshift, fftfreq

from dp_forge.exceptions import ConfigurationError, InvalidMechanismError
from dp_forge.types import ExtractedMechanism

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]


def _logsumexp(log_vals: FloatArray) -> float:
    """Numerically stable log-sum-exp."""
    log_vals = np.asarray(log_vals, dtype=np.float64)
    if len(log_vals) == 0:
        return -np.inf
    m = np.max(log_vals)
    if not np.isfinite(m):
        return m
    return float(m + np.log(np.sum(np.exp(log_vals - m))))


@dataclass
class CharacteristicFunctionResult:
    """
    Result of characteristic function computation.
    
    Attributes:
        cf: Characteristic function values at query points, shape (n_points,)
        query_points: Frequency domain points where CF was evaluated
        log_cf_real: Real part of log(CF) for numerical stability
        log_cf_imag: Imaginary part of log(CF) for numerical stability
        metadata: Optional metadata dict
    """
    cf: ComplexArray
    query_points: FloatArray
    log_cf_real: Optional[FloatArray] = None
    log_cf_imag: Optional[FloatArray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_log_cf(self) -> Tuple[FloatArray, FloatArray]:
        """Convert CF to log-space representation."""
        if self.log_cf_real is not None and self.log_cf_imag is not None:
            return self.log_cf_real, self.log_cf_imag
        
        log_mag = np.log(np.abs(self.cf) + 1e-100)
        phase = np.angle(self.cf)
        
        return log_mag, phase
    
    @staticmethod
    def from_log_cf(
        log_cf_real: FloatArray,
        log_cf_imag: FloatArray,
        query_points: FloatArray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CharacteristicFunctionResult:
        """Create CF result from log-space representation."""
        mag = np.exp(log_cf_real)
        cf = mag * np.exp(1j * log_cf_imag)
        
        return CharacteristicFunctionResult(
            cf=cf,
            query_points=query_points,
            log_cf_real=log_cf_real,
            log_cf_imag=log_cf_imag,
            metadata=metadata or {}
        )


class FourierAccountant:
    """
    Characteristic function based privacy accountant.
    
    Uses FFT to compute tight privacy guarantees via the Fourier transform
    of the privacy loss distribution. Supports heterogeneous composition
    and provides numerical stability via log-space arithmetic.
    
    Attributes:
        grid_size: Number of frequency grid points for FFT
        cf_cache: Cache of computed characteristic functions
        mechanisms: List of added mechanisms
        metadata: Accountant metadata
        
    Example::
    
        accountant = FourierAccountant(grid_size=10000)
        accountant.add_mechanism(mechanism1, adjacent_pair=(0, 1))
        accountant.add_mechanism(mechanism2, adjacent_pair=(1, 2))
        epsilon = accountant.get_epsilon(delta=1e-5)
    """
    
    def __init__(
        self,
        grid_size: int = 10000,
        frequency_range: Optional[Tuple[float, float]] = None,
        use_log_cf: bool = True,
        cache_cfs: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Fourier accountant.
        
        Args:
            grid_size: Number of frequency grid points
            frequency_range: (min_freq, max_freq) or None for auto
            use_log_cf: Use log-CF representation for stability
            cache_cfs: Cache computed CFs for reuse
            metadata: Optional metadata
        """
        if grid_size < 2:
            raise ConfigurationError(
                f"grid_size must be >= 2, got {grid_size}",
                parameter="grid_size"
            )
        
        self.grid_size = grid_size
        self.frequency_range = frequency_range
        self.use_log_cf = use_log_cf
        self.cache_cfs = cache_cfs
        self.metadata = metadata or {}
        
        self.cf_cache: Dict[str, CharacteristicFunctionResult] = {}
        self.mechanisms: List[Dict[str, Any]] = []
        self.composed_cf: Optional[CharacteristicFunctionResult] = None
    
    def add_mechanism(
        self,
        mechanism: Union[ExtractedMechanism, FloatArray],
        adjacent_pair: Optional[Tuple[int, int]] = None,
        name: Optional[str] = None
    ) -> None:
        """
        Add mechanism to composition.
        
        Args:
            mechanism: Mechanism to add
            adjacent_pair: Adjacent database pair (i, i')
            name: Optional mechanism name for caching
        """
        cf_result = characteristic_function(
            mechanism=mechanism,
            adjacent_pair=adjacent_pair,
            grid_size=self.grid_size,
            frequency_range=self.frequency_range,
            use_log_cf=self.use_log_cf
        )
        
        if name and self.cache_cfs:
            self.cf_cache[name] = cf_result
        
        self.mechanisms.append({
            "mechanism": mechanism,
            "adjacent_pair": adjacent_pair,
            "name": name,
            "cf": cf_result
        })
        
        if self.composed_cf is None:
            self.composed_cf = cf_result
        else:
            self.composed_cf = compose_cf(self.composed_cf, cf_result, use_log_cf=self.use_log_cf)
    
    def get_epsilon(self, delta: float) -> float:
        """
        Get epsilon for target delta.
        
        Args:
            delta: Target delta parameter
            
        Returns:
            Epsilon value
        """
        if self.composed_cf is None:
            raise ValueError("No mechanisms added to accountant")
        
        return cf_to_epsilon(
            cf_result=self.composed_cf,
            delta=delta,
            grid_size=self.grid_size
        )
    
    def get_delta(self, epsilon: float) -> float:
        """
        Get delta for target epsilon.
        
        Args:
            epsilon: Target epsilon parameter
            
        Returns:
            Delta value
        """
        if self.composed_cf is None:
            raise ValueError("No mechanisms added to accountant")
        
        return cf_to_delta(
            cf_result=self.composed_cf,
            epsilon=epsilon,
            grid_size=self.grid_size
        )
    
    def reset(self) -> None:
        """Reset accountant state."""
        self.mechanisms.clear()
        self.composed_cf = None
        if not self.cache_cfs:
            self.cf_cache.clear()
    
    def compare_with_rdp(self, delta: float) -> Dict[str, float]:
        """
        Compare Fourier bound with RDP bound.
        
        Args:
            delta: Target delta
            
        Returns:
            Dict with 'fourier_eps', 'rdp_eps', 'improvement'
        """
        from dp_forge.rdp import RDPAccountant
        
        fourier_eps = self.get_epsilon(delta)
        
        rdp_acct = RDPAccountant()
        for mech_info in self.mechanisms:
            mech = mech_info["mechanism"]
            if isinstance(mech, ExtractedMechanism):
                prob_table = mech.probability_table
            else:
                prob_table = mech
            
            pair = mech_info["adjacent_pair"]
            if pair is None:
                continue
            
            i, i_prime = pair
            p_x = prob_table[i] / (np.sum(prob_table[i]) + 1e-100)
            p_x_prime = prob_table[i_prime] / (np.sum(prob_table[i_prime]) + 1e-100)
            
            alphas = np.arange(1.1, 64.0, 0.5)
            rdp_vals = []
            for alpha in alphas:
                rdp = (alpha / (alpha - 1)) * np.log(
                    np.sum(p_x**alpha * p_x_prime**(1 - alpha)) + 1e-100
                )
                rdp_vals.append(rdp)
            
            from dp_forge.rdp import RDPCurve
            curve = RDPCurve(alphas=alphas, epsilons=np.array(rdp_vals))
            rdp_acct.add_rdp_curve(curve)
        
        rdp_eps = rdp_acct.to_dp(delta).epsilon
        
        improvement = (rdp_eps - fourier_eps) / rdp_eps * 100 if rdp_eps > 0 else 0.0
        
        return {
            "fourier_epsilon": fourier_eps,
            "rdp_epsilon": rdp_eps,
            "improvement_percent": improvement,
            "delta": delta
        }


def characteristic_function(
    mechanism: Union[ExtractedMechanism, FloatArray],
    adjacent_pair: Optional[Tuple[int, int]] = None,
    grid_size: int = 10000,
    frequency_range: Optional[Tuple[float, float]] = None,
    use_log_cf: bool = True,
    metadata: Optional[Dict[str, Any]] = None
) -> CharacteristicFunctionResult:
    """
    Compute characteristic function of privacy loss distribution.
    
    The characteristic function is the Fourier transform of the privacy loss PDF:
        φ(t) = E[exp(i·t·L)] = Σ_z P[M(x)=z] · exp(i·t·log(P[M(x)=z]/P[M(x')=z]))
    
    Args:
        mechanism: Mechanism to analyze
        adjacent_pair: Adjacent database pair (i, i')
        grid_size: Number of frequency points
        frequency_range: (min_freq, max_freq) or None for auto
        use_log_cf: Return log-CF representation
        metadata: Optional metadata
        
    Returns:
        CharacteristicFunctionResult with CF values
        
    Notes:
        - CF is computed on a uniform frequency grid
        - Log-CF representation improves numerical stability for composition
        - Auto frequency range: [-4π, 4π] based on privacy loss variance
    """
    if isinstance(mechanism, ExtractedMechanism):
        prob_table = mechanism.probability_table
        if adjacent_pair is None:
            if len(mechanism.adjacencies) == 0:
                raise ValueError("mechanism has no adjacencies and adjacent_pair not provided")
            adjacent_pair = tuple(mechanism.adjacencies[0])
    else:
        prob_table = mechanism
        if adjacent_pair is None:
            raise ValueError("adjacent_pair required when mechanism is array")
    
    i, i_prime = adjacent_pair
    
    p_x = prob_table[i] / (np.sum(prob_table[i]) + 1e-100)
    p_x_prime = prob_table[i_prime] / (np.sum(prob_table[i_prime]) + 1e-100)
    
    epsilon = 1e-100
    log_ratios = np.log(p_x + epsilon) - np.log(p_x_prime + epsilon)
    
    valid_mask = (p_x > epsilon) | (p_x_prime > epsilon)
    log_ratios = log_ratios[valid_mask]
    p_x_valid = p_x[valid_mask]
    
    if frequency_range is None:
        variance = np.sum(p_x_valid * log_ratios**2) - (np.sum(p_x_valid * log_ratios))**2
        std = np.sqrt(variance + epsilon)
        freq_max = 4.0 * np.pi / (std + 0.1)
        frequency_range = (-freq_max, freq_max)
    
    freq_min, freq_max = frequency_range
    query_points = np.linspace(freq_min, freq_max, grid_size)
    
    cf = np.zeros(grid_size, dtype=np.complex128)
    
    for t_idx, t in enumerate(query_points):
        cf[t_idx] = np.sum(p_x_valid * np.exp(1j * t * log_ratios))
    
    result = CharacteristicFunctionResult(
        cf=cf,
        query_points=query_points,
        metadata=metadata or {}
    )
    
    if use_log_cf:
        log_cf_real, log_cf_imag = result.to_log_cf()
        result.log_cf_real = log_cf_real
        result.log_cf_imag = log_cf_imag
    
    return result


def compose_cf(
    cf1: CharacteristicFunctionResult,
    cf2: CharacteristicFunctionResult,
    use_log_cf: bool = True
) -> CharacteristicFunctionResult:
    """
    Compose two characteristic functions via multiplication.
    
    For independent privacy loss random variables L1 and L2:
        φ_{L1+L2}(t) = φ_{L1}(t) · φ_{L2}(t)
    
    Args:
        cf1: First CF
        cf2: Second CF
        use_log_cf: Use log-CF arithmetic for stability
        
    Returns:
        Composed CF
        
    Notes:
        - Query points must match between cf1 and cf2
        - Log-CF composition: log(φ1·φ2) = log(φ1) + log(φ2)
        - Phase unwrapping applied to prevent discontinuities
    """
    if not np.allclose(cf1.query_points, cf2.query_points, rtol=1e-9):
        raise ValueError("CFs must have matching query points for composition")
    
    query_points = cf1.query_points
    
    if use_log_cf and cf1.log_cf_real is not None and cf2.log_cf_real is not None:
        log_cf_real = cf1.log_cf_real + cf2.log_cf_real
        log_cf_imag = cf1.log_cf_imag + cf2.log_cf_imag
        
        log_cf_imag = np.mod(log_cf_imag + np.pi, 2 * np.pi) - np.pi
        
        result = CharacteristicFunctionResult.from_log_cf(
            log_cf_real=log_cf_real,
            log_cf_imag=log_cf_imag,
            query_points=query_points,
            metadata={"composition_of": [cf1.metadata.get("name"), cf2.metadata.get("name")]}
        )
    else:
        composed_cf = cf1.cf * cf2.cf
        
        result = CharacteristicFunctionResult(
            cf=composed_cf,
            query_points=query_points,
            metadata={"composition_of": [cf1.metadata.get("name"), cf2.metadata.get("name")]}
        )
        
        if use_log_cf:
            log_cf_real, log_cf_imag = result.to_log_cf()
            result.log_cf_real = log_cf_real
            result.log_cf_imag = log_cf_imag
    
    return result


def cf_to_epsilon(
    cf_result: CharacteristicFunctionResult,
    delta: float,
    grid_size: Optional[int] = None
) -> float:
    """
    Convert characteristic function to epsilon via numerical inversion.
    
    Uses inverse FFT to recover the privacy loss distribution from its CF,
    then computes the (1-δ)-quantile to determine epsilon.
    
    Args:
        cf_result: CF result from characteristic_function
        delta: Target delta parameter
        grid_size: Optional grid size for inversion (default: use CF grid size)
        
    Returns:
        Epsilon value
        
    Algorithm:
        1. IFFT of CF to get PMF of privacy loss
        2. Integrate PMF from right to find (1-δ)-quantile
        3. Return quantile as epsilon
        
    Complexity:
        O(n log n) for FFT inversion
    """
    if not (0 < delta < 1):
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    
    if grid_size is None:
        grid_size = len(cf_result.cf)
    
    pmf = np.real(ifft(cf_result.cf, n=grid_size))
    pmf = np.maximum(pmf, 0.0)
    
    total_mass = np.sum(pmf)
    if total_mass > 0:
        pmf /= total_mass
    
    freq_step = (cf_result.query_points[-1] - cf_result.query_points[0]) / (len(cf_result.query_points) - 1)
    loss_step = 2.0 * np.pi / (grid_size * freq_step)
    
    cumulative_mass = 0.0
    
    for i in range(grid_size - 1, -1, -1):
        cumulative_mass += pmf[i]
        if cumulative_mass > delta:
            epsilon = (i - grid_size // 2) * loss_step
            return max(0.0, epsilon)
    
    return 0.0


def cf_to_delta(
    cf_result: CharacteristicFunctionResult,
    epsilon: float,
    grid_size: Optional[int] = None
) -> float:
    """
    Convert characteristic function to delta for given epsilon.
    
    Args:
        cf_result: CF result
        epsilon: Target epsilon
        grid_size: Optional grid size
        
    Returns:
        Delta value
    """
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    
    if grid_size is None:
        grid_size = len(cf_result.cf)
    
    pmf = np.real(ifft(cf_result.cf, n=grid_size))
    pmf = np.maximum(pmf, 0.0)
    
    total_mass = np.sum(pmf)
    if total_mass > 0:
        pmf /= total_mass
    
    freq_step = (cf_result.query_points[-1] - cf_result.query_points[0]) / (len(cf_result.query_points) - 1)
    loss_step = 2.0 * np.pi / (grid_size * freq_step)
    
    epsilon_idx = int(epsilon / loss_step) + grid_size // 2
    epsilon_idx = max(0, min(epsilon_idx, grid_size - 1))
    
    delta = np.sum(pmf[epsilon_idx:])
    
    return delta


def batch_compose_cf(
    cf_results: List[CharacteristicFunctionResult],
    use_log_cf: bool = True
) -> CharacteristicFunctionResult:
    """
    Batch compose multiple CFs efficiently.
    
    Composes a list of CFs via vectorized multiplication in frequency domain.
    More efficient than sequential pairwise composition.
    
    Args:
        cf_results: List of CFs to compose
        use_log_cf: Use log-CF arithmetic
        
    Returns:
        Composed CF
        
    Complexity:
        O(m·n) where m = number of CFs, n = grid size
        vs O(m·n·log(n)) for sequential pairwise FFT composition
    """
    if len(cf_results) == 0:
        raise ValueError("cf_results list is empty")
    
    if len(cf_results) == 1:
        return cf_results[0]
    
    query_points = cf_results[0].query_points
    for cf in cf_results[1:]:
        if not np.allclose(query_points, cf.query_points, rtol=1e-9):
            raise ValueError("All CFs must have matching query points")
    
    if use_log_cf and all(cf.log_cf_real is not None for cf in cf_results):
        log_cf_real_sum = np.sum([cf.log_cf_real for cf in cf_results], axis=0)
        log_cf_imag_sum = np.sum([cf.log_cf_imag for cf in cf_results], axis=0)
        
        log_cf_imag_sum = np.mod(log_cf_imag_sum + np.pi, 2 * np.pi) - np.pi
        
        return CharacteristicFunctionResult.from_log_cf(
            log_cf_real=log_cf_real_sum,
            log_cf_imag=log_cf_imag_sum,
            query_points=query_points,
            metadata={"batch_composition": len(cf_results)}
        )
    else:
        cf_array = np.array([cf.cf for cf in cf_results])
        composed_cf = np.prod(cf_array, axis=0)
        
        result = CharacteristicFunctionResult(
            cf=composed_cf,
            query_points=query_points,
            metadata={"batch_composition": len(cf_results)}
        )
        
        if use_log_cf:
            log_cf_real, log_cf_imag = result.to_log_cf()
            result.log_cf_real = log_cf_real
            result.log_cf_imag = log_cf_imag
        
        return result
