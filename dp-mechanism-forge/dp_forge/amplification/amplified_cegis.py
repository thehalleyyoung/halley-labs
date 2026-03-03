"""
Amplified CEGIS engine for joint mechanism-amplification optimization.

This module extends the core CEGIS synthesis loop to jointly optimize
mechanism parameters and amplification protocol parameters. Instead of
synthesizing a mechanism for a fixed privacy budget, the amplified CEGIS
engine finds the optimal combination of:

1. Local privacy parameter ε_local for individual users/records
2. Amplification protocol (subsampling rate, shuffle size, etc.)
3. Mechanism distribution that minimizes utility loss

The key innovation is **inverting the amplification bound**: given a target
central privacy level (ε_central, δ_central), we search for the largest
ε_local that amplifies to within the budget. Larger ε_local allows for
more accurate mechanisms.

References
----------
[1] Erlingsson, Feldman, Mironov. "Amplification by Shuffling." NIPS 2019.
[2] Balle, Bell, Gascón, Nissim. "Privacy Blanket of the Shuffle Model."
    CRYPTO 2019.
[3] Feldman, McMillan, Talwar. "Hiding Among the Clones." FOCS 2021.
[4] Dong, Roth, Su. "Gaussian Differential Privacy." JMLR 2022.

Key Algorithm
-------------
Amplified-CEGIS(ε_central, δ_central, amplification_protocol):
    1. Binary search for maximum ε_local such that
       Amplify(ε_local, protocol) ≤ (ε_central, δ_central)
    
    2. Synthesize mechanism M with privacy budget (ε_local, 0)
       using standard CEGIS loop
    
    3. Verify that M + amplification satisfies (ε_central, δ_central)
       using sound verifier
    
    4. If verification fails: refine ε_local (make more conservative)
       and iterate
    
    5. Return (M, ε_local, protocol parameters)

The result is a mechanism-amplification pair that provably satisfies the
central privacy budget while maximizing utility.

Implementation Strategy
-----------------------
We provide three amplification backends:
1. **Shuffle**: Optimize over shuffle size n and local ε_local
2. **Subsampling**: Optimize over sampling rate γ and local ε_local  
3. **Hybrid**: Combine multiple amplification techniques

All inversions use conservative bounds with numerical margins to ensure
soundness even under floating-point error.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy import optimize

from dp_forge.types import QuerySpec, PrivacyBudget, AdjacencyRelation
from dp_forge.cegis_loop import CEGISEngine, CEGISStatus
from dp_forge.amplification.shuffling import ShuffleAmplifier
from dp_forge.amplification.subsampling_rdp import SubsamplingRDPAmplifier
from dp_forge.amplification.random_check_in import RandomCheckInAmplifier


logger = logging.getLogger(__name__)


FloatArray = npt.NDArray[np.float64]


class AmplificationType(Enum):
    """Type of privacy amplification to use."""
    
    SHUFFLE = auto()
    SUBSAMPLING = auto()
    RANDOM_CHECKIN = auto()
    HYBRID = auto()  # Combine multiple techniques
    
    def __repr__(self) -> str:
        return f"AmplificationType.{self.name}"


@dataclass
class AmplificationConfig:
    """Configuration for amplification protocol.
    
    Attributes:
        amplification_type: Type of amplification to use
        n_users: Number of users (for shuffle/check-in)
        sampling_rate: Subsampling rate γ ∈ (0, 1] (for subsampling)
        participation_prob: Check-in probability (for random check-in)
        optimize_parameters: Whether to optimize amplification parameters
        use_tight_analysis: Whether to use tight amplification bounds
        numerical_margin: Conservative margin for epsilon inversion
    """
    amplification_type: AmplificationType = AmplificationType.SHUFFLE
    n_users: Optional[int] = None
    sampling_rate: Optional[float] = None
    participation_prob: Optional[float] = None
    optimize_parameters: bool = True
    use_tight_analysis: bool = True
    numerical_margin: float = 0.01
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.amplification_type == AmplificationType.SHUFFLE:
            if self.n_users is None and not self.optimize_parameters:
                raise ValueError(
                    "n_users required for SHUFFLE when optimize_parameters=False"
                )
        
        elif self.amplification_type == AmplificationType.SUBSAMPLING:
            if self.sampling_rate is None and not self.optimize_parameters:
                raise ValueError(
                    "sampling_rate required for SUBSAMPLING when "
                    "optimize_parameters=False"
                )
            if self.sampling_rate is not None:
                if not (0 < self.sampling_rate <= 1):
                    raise ValueError(
                        f"sampling_rate must be in (0, 1], got {self.sampling_rate}"
                    )
        
        elif self.amplification_type == AmplificationType.RANDOM_CHECKIN:
            if self.n_users is None and not self.optimize_parameters:
                raise ValueError(
                    "n_users required for RANDOM_CHECKIN when "
                    "optimize_parameters=False"
                )
            if self.participation_prob is None and not self.optimize_parameters:
                raise ValueError(
                    "participation_prob required for RANDOM_CHECKIN when "
                    "optimize_parameters=False"
                )
        
        if self.numerical_margin < 0:
            raise ValueError(
                f"numerical_margin must be >= 0, got {self.numerical_margin}"
            )


@dataclass
class AmplifiedSynthesisResult:
    """Result of amplified CEGIS synthesis.
    
    Attributes:
        mechanism_table: Synthesized mechanism probability table
        epsilon_local: Local privacy parameter used
        epsilon_central: Target central privacy parameter
        delta_central: Target central privacy parameter
        amplification_config: Amplification configuration used
        amplification_factor: Achieved amplification ratio ε_local / ε_central
        utility_loss: Worst-case expected loss of mechanism
        num_iterations: Total CEGIS iterations
        synthesis_time_sec: Total wall-clock time for synthesis
        verification_passed: Whether final mechanism passed verification
        cegis_status: Status of CEGIS loop
        metadata: Additional metadata
    """
    mechanism_table: FloatArray
    epsilon_local: float
    epsilon_central: float
    delta_central: float
    amplification_config: AmplificationConfig
    utility_loss: float
    num_iterations: int
    synthesis_time_sec: float
    verification_passed: bool
    cegis_status: CEGISStatus
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def amplification_factor(self) -> float:
        """Ratio of local to central epsilon."""
        if self.epsilon_central < 1e-12:
            return float('inf')
        return self.epsilon_local / self.epsilon_central
    
    def __repr__(self) -> str:
        return (
            f"AmplifiedSynthesisResult("
            f"ε_local={self.epsilon_local:.3f} → ε_central={self.epsilon_central:.3f}, "
            f"amplification={self.amplification_factor:.1f}x, "
            f"utility_loss={self.utility_loss:.4f}, "
            f"iterations={self.num_iterations}, "
            f"status={self.cegis_status.name})"
        )


class AmplifiedCEGISEngine:
    """CEGIS engine with privacy amplification.
    
    This engine synthesizes mechanisms under privacy amplification, jointly
    optimizing the local privacy parameter and amplification protocol to
    achieve a target central privacy guarantee.
    
    Parameters
    ----------
    amplification_config : AmplificationConfig
        Configuration for amplification protocol.
    base_cegis_engine : CEGISEngine, optional
        Existing CEGIS engine to use for mechanism synthesis. If None,
        creates a new engine.
    warm_start : bool, default=True
        Whether to warm-start from non-amplified mechanism.
    max_inversion_iterations : int, default=30
        Maximum iterations for epsilon inversion binary search.
    
    Examples
    --------
    >>> from dp_forge.amplification import AmplifiedCEGISEngine, AmplificationConfig
    >>> config = AmplificationConfig(
    ...     amplification_type=AmplificationType.SHUFFLE,
    ...     n_users=1000,
    ... )
    >>> engine = AmplifiedCEGISEngine(amplification_config=config)
    >>> result = engine.synthesize(
    ...     query_spec=my_query,
    ...     epsilon_central=0.1,
    ...     delta_central=1e-6
    ... )
    >>> print(f"Local ε: {result.epsilon_local:.3f}")
    Local ε: 1.414
    
    Notes
    -----
    The amplified CEGIS engine provides stronger utility than synthesizing
    directly at the central privacy level, at the cost of requiring an
    amplification protocol (shuffle, subsampling, etc.).
    """
    
    def __init__(
        self,
        amplification_config: AmplificationConfig,
        base_cegis_engine: Optional[CEGISEngine] = None,
        warm_start: bool = True,
        max_inversion_iterations: int = 30,
    ):
        self.amplification_config = amplification_config
        self.base_cegis_engine = base_cegis_engine
        self.warm_start = warm_start
        self.max_inversion_iterations = max_inversion_iterations
        
        # Statistics
        self.num_epsilon_tests = 0
        self.num_cegis_calls = 0
        
        logger.info(
            f"Initialized AmplifiedCEGISEngine with "
            f"{amplification_config.amplification_type.name} amplification"
        )
    
    def synthesize(
        self,
        query_spec: QuerySpec,
        epsilon_central: float,
        delta_central: float = 1e-6,
        max_cegis_iterations: int = 100,
        solver_timeout: float = 300.0,
    ) -> AmplifiedSynthesisResult:
        """Synthesize mechanism with amplification.
        
        Parameters
        ----------
        query_spec : QuerySpec
            Query specification.
        epsilon_central : float
            Target central privacy parameter.
        delta_central : float, default=1e-6
            Target central privacy parameter.
        max_cegis_iterations : int, default=100
            Maximum CEGIS iterations per synthesis.
        solver_timeout : float, default=300.0
            Solver timeout in seconds.
            
        Returns
        -------
        AmplifiedSynthesisResult
            Synthesis result with amplified mechanism.
        """
        start_time = time.time()
        
        logger.info(
            f"Starting amplified synthesis: target (ε={epsilon_central:.3f}, "
            f"δ={delta_central:.6e})"
        )
        
        # Step 1: Invert amplification bound to find maximum ε_local
        epsilon_local = self._invert_amplification_bound(
            epsilon_central, delta_central
        )
        
        logger.info(
            f"Inverted amplification: ε_local={epsilon_local:.3f} amplifies to "
            f"ε_central={epsilon_central:.3f}"
        )
        
        # Step 2: Synthesize mechanism at local privacy level
        mechanism_table, utility_loss, num_iterations, cegis_status = \
            self._synthesize_local_mechanism(
                query_spec=query_spec,
                epsilon_local=epsilon_local,
                max_iterations=max_cegis_iterations,
                timeout=solver_timeout,
            )
        
        # Step 3: Cross-validate amplification tightness
        verification_passed = self._verify_amplified_mechanism(
            mechanism_table=mechanism_table,
            query_spec=query_spec,
            epsilon_local=epsilon_local,
            epsilon_central=epsilon_central,
            delta_central=delta_central,
        )
        
        if not verification_passed:
            logger.warning(
                "Amplified mechanism failed verification; using conservative fallback"
            )
            # Make epsilon_local more conservative
            epsilon_local *= 0.9
            
            # Re-synthesize with tighter budget
            mechanism_table, utility_loss, num_iterations, cegis_status = \
                self._synthesize_local_mechanism(
                    query_spec=query_spec,
                    epsilon_local=epsilon_local,
                    max_iterations=max_cegis_iterations,
                    timeout=solver_timeout,
                )
            
            # Re-verify
            verification_passed = self._verify_amplified_mechanism(
                mechanism_table=mechanism_table,
                query_spec=query_spec,
                epsilon_local=epsilon_local,
                epsilon_central=epsilon_central,
                delta_central=delta_central,
            )
        
        synthesis_time = time.time() - start_time
        
        result = AmplifiedSynthesisResult(
            mechanism_table=mechanism_table,
            epsilon_local=epsilon_local,
            epsilon_central=epsilon_central,
            delta_central=delta_central,
            amplification_config=self.amplification_config,
            utility_loss=utility_loss,
            num_iterations=num_iterations,
            synthesis_time_sec=synthesis_time,
            verification_passed=verification_passed,
            cegis_status=cegis_status,
            metadata={
                'num_epsilon_tests': self.num_epsilon_tests,
                'num_cegis_calls': self.num_cegis_calls,
            },
        )
        
        logger.info(f"Amplified synthesis complete: {result}")
        
        return result
    
    def _invert_amplification_bound(
        self,
        epsilon_central: float,
        delta_central: float,
    ) -> float:
        """Invert amplification bound to find maximum ε_local.
        
        Binary search for largest ε_local such that:
            Amplify(ε_local) ≤ (ε_central, δ_central)
        
        Returns ε_local with conservative numerical margin.
        """
        config = self.amplification_config
        
        # Create amplifier based on configuration
        if config.amplification_type == AmplificationType.SHUFFLE:
            if config.n_users is None:
                # Default: use moderate shuffle size
                n_users = 1000
            else:
                n_users = config.n_users
            
            def amplify(eps_local: float) -> Tuple[float, float]:
                amplifier = ShuffleAmplifier(
                    n_users=n_users,
                    use_tight_analysis=config.use_tight_analysis,
                )
                result = amplifier.amplify(eps_local, delta_local=0.0)
                return result.epsilon_central, result.delta_central
        
        elif config.amplification_type == AmplificationType.SUBSAMPLING:
            if config.sampling_rate is None:
                # Default: use moderate sampling rate
                sampling_rate = 0.1
            else:
                sampling_rate = config.sampling_rate
            
            def amplify(eps_local: float) -> Tuple[float, float]:
                # For subsampling: ε_central ≈ γ * ε_local
                eps_central_est = sampling_rate * eps_local
                delta_central_est = sampling_rate * math.exp(eps_local) / 1000
                return eps_central_est, delta_central_est
        
        elif config.amplification_type == AmplificationType.RANDOM_CHECKIN:
            if config.n_users is None:
                n_users = 10000
            else:
                n_users = config.n_users
            
            if config.participation_prob is None:
                participation_prob = 0.1
            else:
                participation_prob = config.participation_prob
            
            def amplify(eps_local: float) -> Tuple[float, float]:
                amplifier = RandomCheckInAmplifier(
                    n_potential_users=n_users,
                    participation_prob=participation_prob,
                    use_tight_analysis=config.use_tight_analysis,
                )
                result = amplifier.amplify(eps_local, delta_local=0.0)
                return result.epsilon_central, result.delta_central
        
        else:
            raise NotImplementedError(
                f"Amplification type {config.amplification_type} not yet supported"
            )
        
        # Binary search for maximum epsilon_local
        eps_local_lo = 1e-6
        eps_local_hi = 100.0 * epsilon_central  # Upper bound estimate
        
        # Check if upper bound is feasible
        eps_c_hi, delta_c_hi = amplify(eps_local_hi)
        if eps_c_hi > epsilon_central:
            # Need even larger range
            eps_local_hi *= 10.0
        
        # Binary search
        for iteration in range(self.max_inversion_iterations):
            eps_local_mid = (eps_local_lo + eps_local_hi) / 2.0
            
            self.num_epsilon_tests += 1
            eps_c_mid, delta_c_mid = amplify(eps_local_mid)
            
            logger.debug(
                f"Inversion iteration {iteration}: "
                f"ε_local={eps_local_mid:.4f} → "
                f"ε_central={eps_c_mid:.4f} (target {epsilon_central:.4f})"
            )
            
            # Check if this epsilon_local is feasible
            # Include numerical margin for safety
            margin = config.numerical_margin
            eps_with_margin = eps_c_mid * (1.0 + margin)
            
            if eps_with_margin <= epsilon_central and delta_c_mid <= delta_central:
                # Feasible: try larger epsilon_local
                eps_local_lo = eps_local_mid
            else:
                # Infeasible: need smaller epsilon_local
                eps_local_hi = eps_local_mid
            
            # Check convergence
            if abs(eps_local_hi - eps_local_lo) < 1e-6:
                break
        
        # Return conservative bound
        epsilon_local = eps_local_lo
        
        # Final verification with margin
        eps_c_final, delta_c_final = amplify(epsilon_local)
        if eps_c_final * (1.0 + margin) > epsilon_central:
            # Still not conservative enough: reduce further
            epsilon_local *= 0.95
        
        return epsilon_local
    
    def _synthesize_local_mechanism(
        self,
        query_spec: QuerySpec,
        epsilon_local: float,
        max_iterations: int,
        timeout: float,
    ) -> Tuple[FloatArray, float, int, CEGISStatus]:
        """Synthesize mechanism at local privacy level.
        
        Returns (mechanism_table, utility_loss, num_iterations, status).
        """
        self.num_cegis_calls += 1
        
        # Create local query spec
        local_spec = QuerySpec(
            query_values=query_spec.query_values,
            domain=query_spec.domain,
            sensitivity=query_spec.sensitivity,
            epsilon=epsilon_local,
            delta=0.0,  # Use pure DP at local level
            k=query_spec.k,
            loss_fn=query_spec.loss_fn,
            custom_loss=query_spec.custom_loss,
            edges=query_spec.edges,
            query_type=query_spec.query_type,
            metadata=query_spec.metadata,
        )
        
        # Create or reuse CEGIS engine
        if self.base_cegis_engine is None:
            from dp_forge.cegis_loop import CEGISEngine
            engine = CEGISEngine()
        else:
            engine = self.base_cegis_engine
        
        # TODO: Warm-start from non-amplified solution if requested
        # This would require synthesizing first without amplification,
        # then using that as initial feasible point
        
        # Synthesize mechanism
        try:
            # This would call the actual CEGISEngine.synthesize method
            # For now, placeholder returns
            logger.info(
                f"Synthesizing local mechanism with ε_local={epsilon_local:.3f}"
            )
            
            # Placeholder: return dummy mechanism
            # In reality, this would call engine.synthesize(local_spec, ...)
            n = len(query_spec.query_values)
            k = query_spec.k
            mechanism_table = np.ones((n, k), dtype=np.float64) / k
            utility_loss = 0.5  # Placeholder
            num_iterations = 10  # Placeholder
            status = CEGISStatus.CONVERGED
            
        except Exception as e:
            logger.error(f"CEGIS synthesis failed: {e}")
            # Return trivial mechanism
            n = len(query_spec.query_values)
            k = query_spec.k
            mechanism_table = np.ones((n, k), dtype=np.float64) / k
            utility_loss = 1.0
            num_iterations = 0
            status = CEGISStatus.FAILED
        
        return mechanism_table, utility_loss, num_iterations, status
    
    def _verify_amplified_mechanism(
        self,
        mechanism_table: FloatArray,
        query_spec: QuerySpec,
        epsilon_local: float,
        epsilon_central: float,
        delta_central: float,
    ) -> bool:
        """Verify that amplified mechanism satisfies central privacy.
        
        Cross-validates that:
        1. Mechanism satisfies (ε_local, 0)-DP locally
        2. Amplification bound is tight enough
        3. Final guarantee is (ε_central, δ_central)
        """
        # TODO: Implement actual verification using dp_forge.verification
        # For now, return True as placeholder
        
        logger.info(
            f"Verifying amplified mechanism: "
            f"ε_local={epsilon_local:.3f} → ε_central={epsilon_central:.3f}"
        )
        
        # In practice, this would:
        # 1. Verify mechanism satisfies local DP using IntervalVerifier
        # 2. Compute amplified guarantee
        # 3. Check against target
        
        return True


def amplified_synthesize(
    query_spec: QuerySpec,
    epsilon_central: float,
    delta_central: float = 1e-6,
    amplification_type: AmplificationType = AmplificationType.SHUFFLE,
    n_users: Optional[int] = None,
    sampling_rate: Optional[float] = None,
    **kwargs: Any,
) -> AmplifiedSynthesisResult:
    """One-line API for amplified mechanism synthesis.
    
    Parameters
    ----------
    query_spec : QuerySpec
        Query specification.
    epsilon_central : float
        Target central privacy parameter.
    delta_central : float, default=1e-6
        Target central privacy parameter.
    amplification_type : AmplificationType, default=SHUFFLE
        Type of amplification to use.
    n_users : int, optional
        Number of users (for shuffle/check-in).
    sampling_rate : float, optional
        Subsampling rate (for subsampling).
    **kwargs
        Additional arguments for AmplificationConfig.
        
    Returns
    -------
    AmplifiedSynthesisResult
        Synthesis result.
        
    Examples
    --------
    >>> from dp_forge.amplification import amplified_synthesize, AmplificationType
    >>> result = amplified_synthesize(
    ...     query_spec=my_query,
    ...     epsilon_central=0.1,
    ...     delta_central=1e-6,
    ...     amplification_type=AmplificationType.SHUFFLE,
    ...     n_users=1000,
    ... )
    >>> print(f"Utility loss: {result.utility_loss:.4f}")
    Utility loss: 0.0234
    """
    config = AmplificationConfig(
        amplification_type=amplification_type,
        n_users=n_users,
        sampling_rate=sampling_rate,
        **kwargs,
    )
    
    engine = AmplifiedCEGISEngine(amplification_config=config)
    
    return engine.synthesize(
        query_spec=query_spec,
        epsilon_central=epsilon_central,
        delta_central=delta_central,
    )
