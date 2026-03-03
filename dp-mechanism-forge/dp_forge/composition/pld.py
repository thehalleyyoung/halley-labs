"""
Privacy Loss Distribution (PLD) accounting for DP-Forge.

Implements the PLD framework for tight privacy accounting via discretized
representations of privacy loss random variables. Composition is performed
via FFT-based convolution for efficiency. Supports dynamic grid growth,
worst-case PLD computation over adjacent database pairs, and conversion to
(ε, δ)-DP guarantees.

Key Features:
    - Discretized PLD representation on log-scale grid
    - FFT-based convolution for fast composition
    - Worst-case PLD over all adjacent pairs (not per-pair)
    - Dynamic grid growth to prevent FFT aliasing
    - Pessimistic discretization (rounds toward higher privacy cost)
    - Tail truncation with error bounds
    - Log-space arithmetic for numerical stability

References:
    - Koskela, A., Jälkö, J., & Honkela, A. (2020). Computing tight
      differential privacy guarantees using FFT.
    - Doroshenko, V., Ghazi, B., Kamath, G., Kumar, R., & Manurangsi, P.
      (2022). Connect the dots: Tighter discrete approximations of privacy
      loss distributions.

Classes:
    PrivacyLossDistribution — Discretized PLD on log-scale grid

Functions:
    from_mechanism         — Construct PLD from mechanism table
    compose                — Compose two PLDs via FFT convolution
    worst_case_pld         — Compute worst-case PLD over adjacent pairs
    discretize             — Discretize continuous PLD
    to_epsilon_delta       — Convert PLD to (ε, δ) guarantee
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from numpy.fft import fft, ifft, fftshift

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


def _logsumexp_2d(log_vals: FloatArray) -> FloatArray:
    """Vectorized log-sum-exp along last axis."""
    m = np.max(log_vals, axis=-1, keepdims=True)
    return np.squeeze(m + np.log(np.sum(np.exp(log_vals - m), axis=-1, keepdims=True)), axis=-1)


@dataclass
class PrivacyLossDistribution:
    """
    Discretized Privacy Loss Distribution on a log-scale grid.
    
    The PLD represents the distribution of privacy loss L = log(P[M(x)=z] / P[M(x')=z])
    where x and x' are adjacent databases. The distribution is discretized onto
    a uniform grid in log-space for efficient FFT-based composition.
    
    Attributes:
        log_masses: Log-probabilities on grid, shape (grid_size,)
        grid_min: Minimum grid value (privacy loss)
        grid_max: Maximum grid value (privacy loss)
        grid_size: Number of grid points
        tail_mass_upper: Probability mass above grid_max
        tail_mass_lower: Probability mass below grid_min
        metadata: Optional metadata dict
    
    The grid covers [grid_min, grid_max] uniformly with grid_size points.
    Grid spacing: delta = (grid_max - grid_min) / (grid_size - 1).
    
    Privacy loss values below grid_min contribute to tail_mass_lower.
    Privacy loss values above grid_max contribute to tail_mass_upper.
    
    Total probability: sum(exp(log_masses)) + tail_mass_lower + tail_mass_upper = 1
    """
    
    log_masses: FloatArray
    grid_min: float
    grid_max: float
    grid_size: int
    tail_mass_upper: float = 0.0
    tail_mass_lower: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate PLD state."""
        if self.grid_size < 2:
            raise ConfigurationError(
                f"grid_size must be >= 2, got {self.grid_size}",
                parameter="grid_size"
            )
        if self.grid_min >= self.grid_max:
            raise ConfigurationError(
                f"grid_min must be < grid_max, got {self.grid_min} >= {self.grid_max}",
                parameter="grid_min, grid_max"
            )
        if len(self.log_masses) != self.grid_size:
            raise ConfigurationError(
                f"log_masses length {len(self.log_masses)} != grid_size {self.grid_size}",
                parameter="log_masses"
            )
        if self.tail_mass_upper < 0 or self.tail_mass_lower < 0:
            raise ConfigurationError(
                f"tail masses must be non-negative, got upper={self.tail_mass_upper}, lower={self.tail_mass_lower}",
                parameter="tail_mass_upper, tail_mass_lower"
            )
        
        total_mass = np.sum(np.exp(self.log_masses)) + self.tail_mass_upper + self.tail_mass_lower
        if not math.isclose(total_mass, 1.0, rel_tol=1e-6, abs_tol=1e-9):
            warnings.warn(
                f"PLD total mass {total_mass:.10f} deviates from 1.0 by {abs(total_mass - 1.0):.2e}. "
                f"This may indicate numerical issues.",
                RuntimeWarning
            )
    
    @property
    def grid_step(self) -> float:
        """Grid spacing."""
        return (self.grid_max - self.grid_min) / (self.grid_size - 1)
    
    @property
    def grid_values(self) -> FloatArray:
        """Array of grid point values."""
        return np.linspace(self.grid_min, self.grid_max, self.grid_size)
    
    def compose(self, other: PrivacyLossDistribution) -> PrivacyLossDistribution:
        """
        Compose two PLDs via FFT convolution.
        
        Composition corresponds to summing independent privacy loss random variables.
        This is performed via discrete convolution of the probability mass functions.
        
        Args:
            other: Second PLD to compose
            
        Returns:
            Composed PLD
            
        Notes:
            - Grids must be compatible (same grid_step)
            - Result grid spans [self.grid_min + other.grid_min, self.grid_max + other.grid_max]
            - Tail masses are composed pessimistically
        """
        return compose(self, other)
    
    def self_compose(self, count: int) -> PrivacyLossDistribution:
        """
        Self-compose via repeated squaring.
        
        Efficiently computes the composition of `count` independent copies of this PLD
        using the binary representation of `count` and repeated squaring.
        
        Args:
            count: Number of compositions
            
        Returns:
            Self-composed PLD
            
        Complexity:
            O(count * grid_size * log(grid_size)) using FFT convolution
            with repeated squaring reduces the number of FFT operations to O(log(count))
        """
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        if count == 1:
            return self
        
        result = None
        power_of_2_pld = self
        
        while count > 0:
            if count & 1:
                result = power_of_2_pld if result is None else result.compose(power_of_2_pld)
            count >>= 1
            if count > 0:
                power_of_2_pld = power_of_2_pld.compose(power_of_2_pld)
        
        return result
    
    def to_epsilon_delta(self, delta: float) -> float:
        """
        Convert PLD to (ε, δ)-DP guarantee.
        
        Computes the smallest ε such that for all privacy loss L:
            Pr[L > ε] <= δ
            
        This is equivalent to finding the (1-δ)-quantile of the privacy loss distribution.
        
        Args:
            delta: Target delta parameter
            
        Returns:
            Epsilon value
            
        The epsilon is computed by integrating probability mass from the right tail
        until cumulative mass exceeds delta, then solving for the corresponding ε.
        
        Monotonicity property: epsilon is non-decreasing in delta (larger delta allows
        smaller epsilon for the same mechanism).
        """
        if not (0 < delta < 1):
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        
        cumulative_mass = self.tail_mass_upper
        
        if cumulative_mass > delta:
            return self.grid_max + (cumulative_mass - delta) / cumulative_mass * self.grid_step
        
        masses = np.exp(self.log_masses)
        
        for i in range(self.grid_size - 1, -1, -1):
            cumulative_mass += masses[i]
            if cumulative_mass > delta:
                grid_val = self.grid_min + i * self.grid_step
                overshoot_mass = cumulative_mass - delta
                epsilon = grid_val + overshoot_mass / masses[i] * self.grid_step
                return max(0.0, epsilon)
        
        cumulative_mass += self.tail_mass_lower
        if cumulative_mass > delta:
            return max(0.0, self.grid_min)
        
        return 0.0
    
    def to_delta_for_epsilon(self, epsilon: float) -> float:
        """
        Convert PLD to delta for given epsilon.
        
        Computes the smallest δ such that:
            Pr[L > ε] <= δ
            
        Args:
            epsilon: Target epsilon parameter
            
        Returns:
            Delta value
        """
        if epsilon < 0:
            raise ValueError(f"epsilon must be non-negative, got {epsilon}")
        
        if epsilon >= self.grid_max:
            return self.tail_mass_upper
        
        if epsilon <= self.grid_min:
            return 1.0
        
        idx = int((epsilon - self.grid_min) / self.grid_step)
        idx = max(0, min(idx, self.grid_size - 1))
        
        delta = self.tail_mass_upper
        masses = np.exp(self.log_masses)
        
        for i in range(self.grid_size - 1, idx, -1):
            delta += masses[i]
        
        grid_val = self.grid_min + idx * self.grid_step
        if epsilon > grid_val and idx < self.grid_size:
            fraction = (epsilon - grid_val) / self.grid_step
            delta += (1.0 - fraction) * masses[idx]
        
        return delta
    
    def get_epsilon_for_delta(self, delta: float) -> float:
        """Alias for to_epsilon_delta for backward compatibility."""
        return self.to_epsilon_delta(delta)
    
    def get_delta_for_epsilon(self, epsilon: float) -> float:
        """Alias for to_delta_for_epsilon for backward compatibility."""
        return self.to_delta_for_epsilon(epsilon)
    
    def grow_grid(self, new_size: int) -> PrivacyLossDistribution:
        """
        Grow grid to prevent aliasing.
        
        Creates a new PLD with larger grid, preserving the distribution.
        Used when tail masses exceed threshold during composition.
        
        Args:
            new_size: New grid size (must be >= current size)
            
        Returns:
            PLD with grown grid
        """
        if new_size <= self.grid_size:
            return self
        
        old_step = self.grid_step
        new_grid_min = self.grid_min - (new_size - self.grid_size) // 2 * old_step
        new_grid_max = new_grid_min + (new_size - 1) * old_step
        
        new_log_masses = np.full(new_size, -np.inf)
        offset = (new_size - self.grid_size) // 2
        new_log_masses[offset:offset + self.grid_size] = self.log_masses
        
        return PrivacyLossDistribution(
            log_masses=new_log_masses,
            grid_min=new_grid_min,
            grid_max=new_grid_max,
            grid_size=new_size,
            tail_mass_upper=self.tail_mass_upper,
            tail_mass_lower=self.tail_mass_lower,
            metadata=self.metadata.copy()
        )
    
    def truncate_tails(self, tail_bound: float = 1e-15) -> PrivacyLossDistribution:
        """
        Truncate tails with error bounds.
        
        Removes grid points with negligible mass and incorporates them into tail masses.
        
        Args:
            tail_bound: Mass threshold for truncation
            
        Returns:
            Truncated PLD
        """
        masses = np.exp(self.log_masses)
        
        left_idx = 0
        while left_idx < self.grid_size and masses[left_idx] < tail_bound:
            left_idx += 1
        
        right_idx = self.grid_size - 1
        while right_idx >= 0 and masses[right_idx] < tail_bound:
            right_idx -= 1
        
        if left_idx > right_idx:
            return self
        
        new_tail_lower = self.tail_mass_lower + np.sum(masses[:left_idx])
        new_tail_upper = self.tail_mass_upper + np.sum(masses[right_idx + 1:])
        
        new_log_masses = self.log_masses[left_idx:right_idx + 1].copy()
        new_grid_min = self.grid_min + left_idx * self.grid_step
        new_grid_max = self.grid_min + right_idx * self.grid_step
        new_grid_size = right_idx - left_idx + 1
        
        return PrivacyLossDistribution(
            log_masses=new_log_masses,
            grid_min=new_grid_min,
            grid_max=new_grid_max,
            grid_size=new_grid_size,
            tail_mass_upper=new_tail_upper,
            tail_mass_lower=new_tail_lower,
            metadata=self.metadata.copy()
        )


def from_mechanism(
    mechanism: Union[ExtractedMechanism, FloatArray],
    adjacent_pair: Optional[Tuple[int, int]] = None,
    grid_size: int = 10000,
    tail_bound: float = 1e-15,
    metadata: Optional[Dict[str, Any]] = None
) -> PrivacyLossDistribution:
    """
    Construct PLD from mechanism table.
    
    Computes the privacy loss distribution for a given mechanism and adjacent
    database pair. The privacy loss for output z is:
        L_z = log(P[M(x) = z] / P[M(x') = z])
    
    Args:
        mechanism: ExtractedMechanism or probability table of shape (n, k)
        adjacent_pair: Tuple (i, i') of adjacent database indices (required if mechanism is array)
        grid_size: Number of grid points for discretization
        tail_bound: Threshold for tail truncation
        metadata: Optional metadata dict
        
    Returns:
        PrivacyLossDistribution
        
    Notes:
        - Uses pessimistic discretization (rounds toward higher privacy cost)
        - Handles zero probabilities via log-space arithmetic
        - If adjacent_pair is None and mechanism has adjacency info, uses first pair
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
    
    if i < 0 or i >= prob_table.shape[0]:
        raise ValueError(f"Invalid index i={i}, must be in [0, {prob_table.shape[0]})")
    if i_prime < 0 or i_prime >= prob_table.shape[0]:
        raise ValueError(f"Invalid index i'={i_prime}, must be in [0, {prob_table.shape[0]})")
    
    p_x = prob_table[i]
    p_x_prime = prob_table[i_prime]
    
    if not np.all(p_x >= 0) or not np.all(p_x_prime >= 0):
        raise InvalidMechanismError("Probability table contains negative values")
    
    p_x = p_x / np.sum(p_x)
    p_x_prime = p_x_prime / np.sum(p_x_prime)
    
    epsilon = 1e-100
    log_ratios = np.log(p_x + epsilon) - np.log(p_x_prime + epsilon)
    
    valid_mask = (p_x > epsilon) | (p_x_prime > epsilon)
    log_ratios = log_ratios[valid_mask]
    p_x_valid = p_x[valid_mask]
    
    if len(log_ratios) == 0:
        return PrivacyLossDistribution(
            log_masses=np.array([0.0]),
            grid_min=0.0,
            grid_max=0.0,
            grid_size=1,
            tail_mass_upper=0.0,
            tail_mass_lower=0.0,
            metadata=metadata or {}
        )
    
    min_loss = np.min(log_ratios)
    max_loss = np.max(log_ratios)
    
    margin = 0.1 * max(abs(max_loss - min_loss), 0.1)
    grid_min = min_loss - margin
    grid_max = max_loss + margin
    grid_step = (grid_max - grid_min) / (grid_size - 1)
    
    log_masses = np.full(grid_size, -np.inf)
    
    for loss, prob in zip(log_ratios, p_x_valid):
        if prob < epsilon:
            continue
        idx = int((loss - grid_min) / grid_step)
        idx = max(0, min(idx, grid_size - 1))
        
        if log_masses[idx] == -np.inf:
            log_masses[idx] = np.log(prob)
        else:
            log_masses[idx] = np.logaddexp(log_masses[idx], np.log(prob))
    
    return PrivacyLossDistribution(
        log_masses=log_masses,
        grid_min=grid_min,
        grid_max=grid_max,
        grid_size=grid_size,
        tail_mass_upper=0.0,
        tail_mass_lower=0.0,
        metadata=metadata or {}
    )


def compose(pld1: PrivacyLossDistribution, pld2: PrivacyLossDistribution) -> PrivacyLossDistribution:
    """
    Compose two PLDs via FFT convolution.
    
    The composition of two mechanisms corresponds to summing their privacy loss
    random variables. This is computed via discrete convolution of the PMFs.
    
    Args:
        pld1: First PLD
        pld2: Second PLD
        
    Returns:
        Composed PLD
        
    Algorithm:
        1. Align grids to common grid_step
        2. Pad grids to cover composed range
        3. FFT both PMFs
        4. Multiply in frequency domain
        5. IFFT to get composed PMF
        6. Compose tail masses pessimistically
        
    Complexity:
        O(n log n) where n = grid_size
    """
    if not math.isclose(pld1.grid_step, pld2.grid_step, rel_tol=1e-6):
        target_step = min(pld1.grid_step, pld2.grid_step)
        
        if pld1.grid_step > target_step * 1.01:
            pld1 = _regrid_pld(pld1, target_step)
        if pld2.grid_step > target_step * 1.01:
            pld2 = _regrid_pld(pld2, target_step)
    
    grid_step = pld1.grid_step
    composed_min = pld1.grid_min + pld2.grid_min
    composed_max = pld1.grid_max + pld2.grid_max
    composed_size = int((composed_max - composed_min) / grid_step) + 1
    
    fft_size = 1 << (composed_size - 1).bit_length()
    
    masses1 = np.exp(pld1.log_masses)
    masses2 = np.exp(pld2.log_masses)
    
    pmf1 = np.zeros(fft_size)
    pmf2 = np.zeros(fft_size)
    
    offset1 = int((pld1.grid_min - composed_min) / grid_step)
    offset2 = int((pld2.grid_min - composed_min) / grid_step)
    
    pmf1[offset1:offset1 + pld1.grid_size] = masses1
    pmf2[offset2:offset2 + pld2.grid_size] = masses2
    
    fft1 = fft(pmf1)
    fft2 = fft(pmf2)
    composed_fft = fft1 * fft2
    composed_pmf = np.real(ifft(composed_fft))
    
    composed_pmf = composed_pmf[:composed_size]
    composed_pmf = np.maximum(composed_pmf, 0.0)
    
    epsilon = 1e-100
    composed_log_masses = np.log(composed_pmf + epsilon)
    composed_log_masses[composed_pmf < epsilon] = -np.inf
    
    composed_tail_upper = pld1.tail_mass_upper + pld2.tail_mass_upper
    composed_tail_lower = pld1.tail_mass_lower + pld2.tail_mass_lower
    
    result = PrivacyLossDistribution(
        log_masses=composed_log_masses,
        grid_min=composed_min,
        grid_max=composed_max,
        grid_size=composed_size,
        tail_mass_upper=composed_tail_upper,
        tail_mass_lower=composed_tail_lower,
        metadata={"composition_of": [pld1.metadata.get("name"), pld2.metadata.get("name")]}
    )
    
    if result.tail_mass_upper > 1e-12 or result.tail_mass_lower > 1e-12:
        warnings.warn(
            f"Tail masses after composition: upper={result.tail_mass_upper:.2e}, "
            f"lower={result.tail_mass_lower:.2e}. Consider growing grid.",
            RuntimeWarning
        )
    
    return result


def _regrid_pld(pld: PrivacyLossDistribution, target_step: float) -> PrivacyLossDistribution:
    """Regrid PLD to target grid step."""
    new_size = int((pld.grid_max - pld.grid_min) / target_step) + 1
    new_grid = np.linspace(pld.grid_min, pld.grid_max, new_size)
    
    masses = np.exp(pld.log_masses)
    old_grid = pld.grid_values
    
    new_masses = np.interp(new_grid, old_grid, masses, left=0.0, right=0.0)
    
    total_mass = np.sum(new_masses)
    if total_mass > 0:
        new_masses /= total_mass
    
    epsilon = 1e-100
    new_log_masses = np.log(new_masses + epsilon)
    new_log_masses[new_masses < epsilon] = -np.inf
    
    return PrivacyLossDistribution(
        log_masses=new_log_masses,
        grid_min=pld.grid_min,
        grid_max=pld.grid_max,
        grid_size=new_size,
        tail_mass_upper=pld.tail_mass_upper,
        tail_mass_lower=pld.tail_mass_lower,
        metadata=pld.metadata.copy()
    )


def worst_case_pld(
    mechanism: Union[ExtractedMechanism, FloatArray],
    adjacencies: Optional[List[Tuple[int, int]]] = None,
    grid_size: int = 10000,
    tail_bound: float = 1e-15,
    metadata: Optional[Dict[str, Any]] = None
) -> PrivacyLossDistribution:
    """
    Compute worst-case PLD over all adjacent pairs.
    
    Instead of computing per-pair PLDs, computes a single PLD that upper-bounds
    the privacy loss across all adjacent database pairs. This is the PLD of the
    worst-case adjacent pair (the pair with maximum privacy loss).
    
    Args:
        mechanism: ExtractedMechanism or probability table
        adjacencies: List of (i, i') adjacent pairs (optional if mechanism has adjacencies)
        grid_size: Grid size for discretization
        tail_bound: Tail truncation threshold
        metadata: Optional metadata
        
    Returns:
        Worst-case PLD
        
    Notes:
        - Performance critical: O(|adjacencies| * k) not O(|adjacencies| * k * grid_size)
        - Identifies worst-case pair by maximum KL divergence
        - Constructs PLD only for that pair
    """
    if isinstance(mechanism, ExtractedMechanism):
        prob_table = mechanism.probability_table
        if adjacencies is None:
            adjacencies = [tuple(adj) for adj in mechanism.adjacencies]
    else:
        prob_table = mechanism
        if adjacencies is None:
            raise ValueError("adjacencies required when mechanism is array")
    
    if len(adjacencies) == 0:
        raise ValueError("No adjacencies provided")
    
    worst_pair = None
    worst_kl = -np.inf
    epsilon = 1e-100
    
    for pair in adjacencies:
        i, i_prime = pair
        p_x = prob_table[i] / (np.sum(prob_table[i]) + epsilon)
        p_x_prime = prob_table[i_prime] / (np.sum(prob_table[i_prime]) + epsilon)
        
        kl = np.sum(p_x * np.log((p_x + epsilon) / (p_x_prime + epsilon)))
        
        if kl > worst_kl:
            worst_kl = kl
            worst_pair = pair
    
    return from_mechanism(
        mechanism=mechanism,
        adjacent_pair=worst_pair,
        grid_size=grid_size,
        tail_bound=tail_bound,
        metadata={**(metadata or {}), "worst_pair": worst_pair, "worst_kl": worst_kl}
    )


def discretize(
    continuous_pld: Callable[[float], float],
    grid_size: int,
    grid_min: float,
    grid_max: float,
    tail_bound: float = 1e-15,
    pessimistic: bool = True,
    metadata: Optional[Dict[str, Any]] = None
) -> PrivacyLossDistribution:
    """
    Discretize continuous PLD onto grid.
    
    Converts a continuous privacy loss probability density function to a
    discretized PLD suitable for FFT-based composition.
    
    Args:
        continuous_pld: Function mapping privacy loss -> probability density
        grid_size: Number of grid points
        grid_min: Minimum grid value
        grid_max: Maximum grid value
        tail_bound: Tail truncation threshold
        pessimistic: If True, round toward higher privacy cost
        metadata: Optional metadata
        
    Returns:
        Discretized PLD
        
    Notes:
        - Uses trapezoidal rule for integration over grid cells
        - Pessimistic mode rounds up probabilities (conservative)
        - Tail masses computed via numerical integration outside grid bounds
    """
    if grid_size < 2:
        raise ValueError(f"grid_size must be >= 2, got {grid_size}")
    if grid_min >= grid_max:
        raise ValueError(f"grid_min must be < grid_max, got {grid_min} >= {grid_max}")
    
    grid_step = (grid_max - grid_min) / (grid_size - 1)
    grid_vals = np.linspace(grid_min, grid_max, grid_size)
    
    masses = np.zeros(grid_size)
    
    for i in range(grid_size):
        left = grid_vals[i] - grid_step / 2 if i > 0 else -np.inf
        right = grid_vals[i] + grid_step / 2 if i < grid_size - 1 else np.inf
        
        if i == 0:
            masses[i] = continuous_pld(grid_vals[i]) * (grid_step / 2)
        elif i == grid_size - 1:
            masses[i] = continuous_pld(grid_vals[i]) * (grid_step / 2)
        else:
            masses[i] = continuous_pld(grid_vals[i]) * grid_step
        
        if pessimistic:
            masses[i] *= 1.001
    
    masses = np.maximum(masses, 0.0)
    total_mass = np.sum(masses)
    
    if total_mass > 0:
        masses /= total_mass
    
    epsilon = 1e-100
    log_masses = np.log(masses + epsilon)
    log_masses[masses < epsilon] = -np.inf
    
    return PrivacyLossDistribution(
        log_masses=log_masses,
        grid_min=grid_min,
        grid_max=grid_max,
        grid_size=grid_size,
        tail_mass_upper=0.0,
        tail_mass_lower=0.0,
        metadata=metadata or {}
    )


def to_epsilon_delta(pld: PrivacyLossDistribution, delta: float) -> float:
    """
    Convert PLD to (ε, δ) guarantee.
    
    Convenience function that delegates to pld.to_epsilon_delta(delta).
    
    Args:
        pld: Privacy loss distribution
        delta: Target delta parameter
        
    Returns:
        Epsilon value
    """
    return pld.to_epsilon_delta(delta)
